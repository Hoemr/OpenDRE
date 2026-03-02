# -*- coding: utf-8 -*-
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from sde import ImageInterpXt
from pl_data import OOD_EVAL_CONFIG
from .base import DensityRatioEstimationModel

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
        ood_names = list(OOD_EVAL_CONFIG[self.ood_in_dist]["ood_datasets"])
        batch_size = getattr(self.args, "test_batch_size", self.args.batch_size)
        id_loader = DataLoader(id_dataset, batch_size=batch_size, shuffle=False)
        id_scores = self._compute_scores_for_loader(id_loader)
        all_aurocs = []
        for ood_name, ood_ds in zip(ood_names, ood_datasets):
            ood_loader = DataLoader(ood_ds, batch_size=batch_size, shuffle=False)
            ood_scores = self._compute_scores_for_loader(ood_loader)
            labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
            scores = np.concatenate([id_scores, ood_scores])
            try:
                auroc = roc_auc_score(labels, scores)
                self.log("val_auroc_%s" % ood_name, auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                all_aurocs.append(auroc)
            except Exception:
                self.log("val_auroc_%s" % ood_name, float("nan"), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if all_aurocs:
            self.log("val_auroc_mean", np.mean(all_aurocs), on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        return {}

    def on_test_epoch_end(self):
        id_dataset = self.datamodule._get_in_dist_dataset(train=True)
        ood_datasets = self.datamodule.get_eval_ood_datasets(stage="test")
        ood_names = list(OOD_EVAL_CONFIG[self.ood_in_dist]["ood_datasets"])
        id_loader = DataLoader(id_dataset, batch_size=self.args.test_batch_size, shuffle=False)
        id_scores = self._compute_scores_for_loader(id_loader)
        all_aurocs = []
        for ood_name, ood_ds in zip(ood_names, ood_datasets):
            ood_loader = DataLoader(ood_ds, batch_size=self.args.test_batch_size, shuffle=False)
            ood_scores = self._compute_scores_for_loader(ood_loader)
            labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
            scores = np.concatenate([id_scores, ood_scores])
            try:
                auroc = roc_auc_score(labels, scores)
                self.log("test_auroc_%s" % ood_name, auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                all_aurocs.append(auroc)
            except Exception:
                self.log("test_auroc_%s" % ood_name, float("nan"), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if all_aurocs:
            self.log("test_auroc_mean", np.mean(all_aurocs), on_step=False, on_epoch=True, prog_bar=True, logger=True)

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
