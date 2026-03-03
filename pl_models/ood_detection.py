# -*- coding: utf-8 -*-
import logging
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from path import ImageInterpXt
from pl_data import OOD_EVAL_CONFIG
from .base import DensityRatioEstimationModel

_TRAIN_LOGGER_NAME = "training_logger"


def _ood_metrics(labels, scores):
    """
    Compute OpenOOD-style metrics. labels: 1 = ID, 0 = OOD; scores: higher = more ID-like.
    Returns dict with auroc, fpr95, aupr_in, aupr_out, acc.
    """
    if np.sum(labels) == 0 or np.sum(1 - labels) == 0:
        return {"auroc": np.nan, "fpr95": np.nan, "aupr_in": np.nan, "aupr_out": np.nan, "acc": np.nan}
    id_scores = scores[labels == 1]
    ood_scores = scores[labels == 0]
    thresh = np.percentile(id_scores, 5.0)  # 95% of ID above this -> TPR 95%
    fpr95 = float(np.mean(ood_scores >= thresh))
    pred = (scores >= thresh).astype(np.float64)
    acc = float(np.mean(pred == labels))
    try:
        auroc = float(roc_auc_score(labels, scores))
    except Exception:
        auroc = np.nan
    try:
        aupr_in = float(average_precision_score(labels, scores))   # ID as positive
    except Exception:
        aupr_in = np.nan
    try:
        aupr_out = float(average_precision_score(1 - labels, -np.array(scores)))  # OOD as positive
    except Exception:
        aupr_out = np.nan
    return {"auroc": auroc, "fpr95": fpr95, "aupr_in": aupr_in, "aupr_out": aupr_out, "acc": acc}

class OutOfDistributionDetectionModel(DensityRatioEstimationModel):
    def __init__(self, args, save_path, datamodule):
        super().__init__(args, save_path, datamodule)
        self.ood_in_dist = getattr(args, "ood_in_dist", "mnist")
        self.ood_base_type = getattr(args, "ood_base_type", "universal")
        self.train_loss_sum = 0.0
        self.train_loss_count = 0

    def _build_sde(self, args):
        return ImageInterpXt(args=args)

    def training_step(self, batch, batch_idx):
        px, qx, _ = self.prepare_batch(batch)
        loss_dict = self.train_step_fn(self.model, (qx, px), step=self.global_step)
        self.train_loss_sum += loss_dict["loss"].item() * px.size(0)
        self.train_loss_count += px.size(0)
        return loss_dict["loss"]

    def on_train_epoch_end(self):
        avg_train_loss = self.train_loss_sum / max(self.train_loss_count, 1)
        self.log("train_loss", avg_train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.train_loss_sum = 0.0
        self.train_loss_count = 0
        super().on_train_epoch_end()

    def compute_ood_scores(self, batch):
        with torch.no_grad():
            log_ratio, _ = self.density_ratio_fn(self.model, batch, joint=None)
            return log_ratio.squeeze()

    def validation_step(self, batch, batch_idx):
        return {}

    def on_validation_epoch_end(self):
        id_dataset = self.datamodule.train_dataset.in_dist_dataset
        ood_datasets = self.datamodule.get_eval_ood_datasets(stage="val")
        ood_names = getattr(self.datamodule, "eval_ood_names", None) or list(OOD_EVAL_CONFIG[self.ood_in_dist]["ood_datasets"])
        batch_size = getattr(self.args, "test_batch_size", self.args.batch_size)
        id_loader = DataLoader(id_dataset, batch_size=batch_size, shuffle=False)
        id_scores = self._compute_scores_for_loader(id_loader)
        all_aurocs, all_fpr95, all_aupr_in, all_aupr_out, all_acc = [], [], [], [], []
        for ood_name, ood_ds in zip(ood_names, ood_datasets):
            ood_loader = DataLoader(ood_ds, batch_size=batch_size, shuffle=False)
            ood_scores = self._compute_scores_for_loader(ood_loader)
            labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
            scores = np.concatenate([id_scores, ood_scores])
            m = _ood_metrics(labels, scores)
            for key, val in m.items():
                self.log("val_%s_%s" % (key, ood_name), val, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            all_aurocs.append(m["auroc"] if not np.isnan(m["auroc"]) else None)
            all_fpr95.append(m["fpr95"] if not np.isnan(m["fpr95"]) else None)
            all_aupr_in.append(m["aupr_in"] if not np.isnan(m["aupr_in"]) else None)
            all_aupr_out.append(m["aupr_out"] if not np.isnan(m["aupr_out"]) else None)
            all_acc.append(m["acc"] if not np.isnan(m["acc"]) else None)
        mean_metrics = {}
        for name, vals in [("auroc", all_aurocs), ("fpr95", all_fpr95), ("aupr_in", all_aupr_in), ("aupr_out", all_aupr_out), ("acc", all_acc)]:
            valid = [v for v in vals if v is not None]
            mean_val = np.mean(valid) if valid else float("nan")
            self.log("val_%s_mean" % name, mean_val, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            mean_metrics[name] = mean_val
        self._log_ood_val_to_txt(mean_metrics, ood_names, all_aurocs)

    def _log_ood_val_to_txt(self, mean_metrics, ood_names, all_aurocs):
        """Write validation metrics to training_logger so they appear even when callback runs before model (Lightning v2)."""
        if not (self.trainer and self.trainer.is_global_zero):
            return
        log = logging.getLogger(_TRAIN_LOGGER_NAME)
        if not log.handlers:
            return
        epoch = self.trainer.current_epoch + 1
        parts = [f"Epoch {epoch}/{self.trainer.max_epochs}"]
        for k, v in [("AUROC", mean_metrics.get("auroc")), ("FPR95", mean_metrics.get("fpr95")),
                     ("AUPR-IN", mean_metrics.get("aupr_in")), ("AUPR-OUT", mean_metrics.get("aupr_out")), ("ACC", mean_metrics.get("acc"))]:
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                parts.append(f"{k} {v:.4f}")
            else:
                parts.append(f"{k} nan")
        for name, auroc in zip(ood_names, all_aurocs):
            if auroc is not None:
                parts.append(f"{name} {auroc:.4f}")
        log.info(", ".join(parts))

    def test_step(self, batch, batch_idx):
        return {}

    def on_test_epoch_end(self):
        id_dataset = self.datamodule._get_in_dist_dataset(train=True)
        ood_datasets = self.datamodule.get_eval_ood_datasets(stage="test")
        ood_names = getattr(self.datamodule, "eval_ood_names", None) or list(OOD_EVAL_CONFIG[self.ood_in_dist]["ood_datasets"])
        batch_size = getattr(self.args, "test_batch_size", self.args.batch_size)
        id_loader = DataLoader(id_dataset, batch_size=batch_size, shuffle=False)
        id_scores = self._compute_scores_for_loader(id_loader)
        all_aurocs, all_fpr95, all_aupr_in, all_aupr_out, all_acc = [], [], [], [], []
        for ood_name, ood_ds in zip(ood_names, ood_datasets):
            ood_loader = DataLoader(ood_ds, batch_size=batch_size, shuffle=False)
            ood_scores = self._compute_scores_for_loader(ood_loader)
            labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
            scores = np.concatenate([id_scores, ood_scores])
            m = _ood_metrics(labels, scores)
            for key, val in m.items():
                self.log("test_%s_%s" % (key, ood_name), val, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            all_aurocs.append(m["auroc"] if not np.isnan(m["auroc"]) else None)
            all_fpr95.append(m["fpr95"] if not np.isnan(m["fpr95"]) else None)
            all_aupr_in.append(m["aupr_in"] if not np.isnan(m["aupr_in"]) else None)
            all_aupr_out.append(m["aupr_out"] if not np.isnan(m["aupr_out"]) else None)
            all_acc.append(m["acc"] if not np.isnan(m["acc"]) else None)
        mean_metrics = {}
        for name, vals in [("auroc", all_aurocs), ("fpr95", all_fpr95), ("aupr_in", all_aupr_in), ("aupr_out", all_aupr_out), ("acc", all_acc)]:
            valid = [v for v in vals if v is not None]
            mean_val = np.mean(valid) if valid else float("nan")
            self.log("test_%s_mean" % name, mean_val, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            mean_metrics[name] = mean_val
        self._log_ood_test_to_txt(mean_metrics, ood_names, all_aurocs)

    def _log_ood_test_to_txt(self, mean_metrics, ood_names, all_aurocs):
        """Write test metrics to training_logger (Lightning v2 callback order)."""
        if not (self.trainer and self.trainer.is_global_zero):
            return
        log = logging.getLogger(_TRAIN_LOGGER_NAME)
        if not log.handlers:
            return
        parts = ["Test Epoch"]
        for k, v in [("AUROC", mean_metrics.get("auroc")), ("FPR95", mean_metrics.get("fpr95")),
                     ("AUPR-IN", mean_metrics.get("aupr_in")), ("AUPR-OUT", mean_metrics.get("aupr_out")), ("ACC", mean_metrics.get("acc"))]:
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                parts.append(f"{k} {v:.4f}")
            else:
                parts.append(f"{k} nan")
        for name, auroc in zip(ood_names, all_aurocs):
            if auroc is not None:
                parts.append(f"{name} {auroc:.4f}")
        log.info(", ".join(parts))

    def _compute_scores_for_loader(self, loader):
        self.model.eval()
        all_scores = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device)
                scores = self.compute_ood_scores(x)
                all_scores.append(scores.cpu().numpy())
        return np.concatenate(all_scores)
