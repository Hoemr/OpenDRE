# -*- coding: utf-8 -*-
import logging
import os
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from dre.path import ImageInterpXt
from pl_data import OOD_EVAL_CONFIG, OOD_IMGLIST_NEAR_FAR
from .base import DensityRatioEstimationModel
import utils.visualization as vis_utils

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
    # thresh = np.percentile(id_scores, 5.0)  # 95% of ID above this -> TPR 95%
    thresh = np.percentile(id_scores, 5.0, interpolation='higher')
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

    # def configure_optimizers(self):
    #     """OpenOOD-style SGD optimizer with momentum."""
    #     optimizer = torch.optim.SGD(
    #         self.model.parameters(),
    #         lr=getattr(self.args, "lr", 0.1),
    #         momentum=getattr(self.args, "momentum", 0.9),
    #         weight_decay=getattr(self.args, "weight_decay", 5e-4),
    #     )
    #     return {"optimizer": optimizer}

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
        self._evaluate_ood(stage="val")
        vis_utils.visualize_path_function(
            self.path, sample_path=self.sample_path,
            path_type=self.args.path_type,
            constraint_type=self.args.constraint_type
        )
        vis_utils.visualize_variance_profiles(
            self.path, sample_path=self.sample_path,
            path_type=self.args.path_type
        )

    def test_step(self, batch, batch_idx):
        return {}

    def on_test_epoch_end(self):
        self._evaluate_ood(stage="test")

    def _evaluate_ood(self, stage="val"):
        """Unified OOD evaluation for validation and test stages.
        
        IMPORTANT: Uses separate ID splits for evaluation to avoid data leakage:
        - validation: uses ID val split (val_cifar10.txt)
        - test: uses ID test split (test_cifar10.txt)
        
        This ensures the model is evaluated on data NOT seen during training.
        """
        # Use correct ID split for evaluation (NOT training data)
        id_dataset = self.datamodule.get_eval_id_dataset(stage=stage)
        ood_datasets = self.datamodule.get_eval_ood_datasets(stage=stage)
        ood_names = getattr(self.datamodule, "eval_ood_names", None) or list(OOD_EVAL_CONFIG[self.ood_in_dist]["ood_datasets"])
        batch_size = getattr(self.args, "test_batch_size", self.args.batch_size)
        prefix = "val" if stage == "val" else "test"

        id_loader = DataLoader(id_dataset, batch_size=batch_size, shuffle=False)
        eval_time_sec = None
        score_time_sec = 0.0

        # Compute ID scores (timed for both val and test)
        t0 = time.perf_counter()
        id_scores = self._compute_scores_for_loader(id_loader)
        score_time_sec += time.perf_counter() - t0

        # Collect metrics and scores per OOD dataset
        all_metrics = {"auroc": [], "fpr95": [], "aupr_in": [], "aupr_out": [], "acc": []}
        all_ood_scores = {}
        for ood_name, ood_ds in zip(ood_names, ood_datasets):
            ood_loader = DataLoader(ood_ds, batch_size=batch_size, shuffle=False)
            t1 = time.perf_counter()
            ood_scores = self._compute_scores_for_loader(ood_loader)
            score_time_sec += time.perf_counter() - t1
            all_ood_scores[ood_name] = ood_scores
            labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
            scores = np.concatenate([id_scores, ood_scores])
            m = _ood_metrics(labels, scores)
            for key, val in m.items():
                self.log(f"{prefix}_{key}_{ood_name}", val, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                all_metrics[key].append(val if not np.isnan(val) else None)

        # Log inference time as a metric (ID + all OOD scores)
        eval_time_sec = score_time_sec
        self.log(
            f"{prefix}_time",
            eval_time_sec,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        # Save raw scores for further analysis (per run, per stage)
        try:
            metrics_dir = self.metrics_dir
            os.makedirs(metrics_dir, exist_ok=True)
            save_dict = {"id": id_scores}
            for ood_name, ood_scores in all_ood_scores.items():
                save_dict[f"ood_{ood_name}"] = ood_scores
            np.savez_compressed(os.path.join(metrics_dir, f"{stage}_scores.npz"), **save_dict)
        except Exception as e:
            # Do not break training/eval if saving fails, but surface the error once
            print(f"[Warning] Failed to save OOD scores to metrics/{stage}_scores.npz: {e}")

        # Compute mean metrics
        mean_metrics = {}
        for name in all_metrics:
            valid = [v for v in all_metrics[name] if v is not None]
            mean_val = np.mean(valid) if valid else float("nan")
            self.log(f"{prefix}_{name}_mean", mean_val, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            mean_metrics[name] = mean_val

        # Compute near/far AUROC
        name_to_auroc = {n: a for n, a in zip(ood_names, all_metrics["auroc"]) if a is not None}
        near_auroc, far_auroc = self._compute_near_far_auroc(name_to_auroc, prefix)

        # Log to txt file (include eval time for both val and test stages)
        self._log_ood_to_txt(
            mean_metrics,
            ood_names,
            all_metrics["auroc"],
            near_auroc,
            far_auroc,
            is_val=(stage == "val"),
            eval_time_sec=eval_time_sec,
        )

    def _compute_near_far_auroc(self, name_to_auroc, prefix):
        """Compute and log near/far AUROC averages."""
        near_far = OOD_IMGLIST_NEAR_FAR.get(self.ood_in_dist, {}) if getattr(self.args, "ood_use_imglist", False) else {}
        near_auroc, far_auroc = None, None
        if near_far:
            near_vals = [name_to_auroc[n] for n in near_far.get("near", []) if n in name_to_auroc]
            far_vals = [name_to_auroc[n] for n in near_far.get("far", []) if n in name_to_auroc]
            if near_vals:
                near_auroc = float(np.mean(near_vals))
                self.log(f"{prefix}_near_auroc", near_auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            if far_vals:
                far_auroc = float(np.mean(far_vals))
                self.log(f"{prefix}_far_auroc", far_auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return near_auroc, far_auroc

    def _log_ood_to_txt(self, mean_metrics, ood_names, all_aurocs, near_auroc, far_auroc, is_val=True, eval_time_sec=None):
        """Write OOD metrics to training_logger. Group by near-OOD / far-OOD when using imglist."""
        if not (self.trainer and self.trainer.is_global_zero):
            return
        log = logging.getLogger(_TRAIN_LOGGER_NAME)
        if not log.handlers:
            return

        # Build log line
        if is_val:
            epoch = self.trainer.current_epoch + 1
            parts = [f"Epoch {epoch}/{self.trainer.max_epochs}"]
        else:
            parts = ["Test"]

        # Mean metrics
        for k, v in [("AUROC", mean_metrics.get("auroc")), ("FPR95", mean_metrics.get("fpr95")),
                     ("AUPR-IN", mean_metrics.get("aupr_in")), ("AUPR-OUT", mean_metrics.get("aupr_out")), ("ACC", mean_metrics.get("acc"))]:
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                parts.append(f"{k} {v:.4f}")
            else:
                parts.append(f"{k} nan")

        # near/far AUROC
        if near_auroc is not None:
            parts.append(f"near-AUROC {near_auroc:.4f}")
        if far_auroc is not None:
            parts.append(f"far-AUROC {far_auroc:.4f}")

        # Per-dataset AUROC
        name_to_auroc = {n: a for n, a in zip(ood_names, all_aurocs) if a is not None}
        near_far = OOD_IMGLIST_NEAR_FAR.get(self.ood_in_dist, {}) if getattr(self.args, "ood_use_imglist", False) else {}
        if near_far:
            near_parts = [f"{n} {name_to_auroc[n]:.4f}" for n in near_far.get("near", []) if n in name_to_auroc]
            far_parts = [f"{n} {name_to_auroc[n]:.4f}" for n in near_far.get("far", []) if n in name_to_auroc]
            if near_parts:
                parts.append("near-OOD: " + ", ".join(near_parts))
            if far_parts:
                parts.append("far-OOD: " + ", ".join(far_parts))
        else:
            for name, auroc in zip(ood_names, all_aurocs):
                if auroc is not None:
                    parts.append(f"{name} {auroc:.4f}")

        # Append inference time (seconds)
        if eval_time_sec is not None:
            if is_val:
                parts.append(f"val_time {eval_time_sec:.2f}s")
            else:
                parts.append(f"test_time {eval_time_sec:.2f}s")

        log.info(" | ".join(parts))

    def _compute_scores_for_loader(self, loader):
        self.model.eval()
        all_scores = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device)
                # Adapt channel count: if single channel (e.g., MNIST), replicate to 3 channels
                if x.shape[1] == 1:
                    x = x.repeat(1, 3, 1, 1)
                scores = self.compute_ood_scores(x)
                all_scores.append(scores.cpu().numpy())
        return np.concatenate(all_scores)