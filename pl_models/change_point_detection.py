# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn.functional as F
import core.visualization as vis_utils
from .base import DensityRatioEstimationModel

class ChangePointDetectionModel(DensityRatioEstimationModel):
    def __init__(self, args, save_path, datamodule):
        super().__init__(args, save_path, datamodule)
        self.swl = getattr(args, "cpd_swl", 64)
        self.single_test = getattr(args, "single_test", False)
        self.use_simplex = getattr(args, "use_simplex", False)
        self.test_step_outputs = []
        self.validation_step_outputs = []
        self.preprocess = (lambda x: F.softmax(x, dim=-1)) if self.use_simplex else (lambda x: x)

    def training_step(self, batch, batch_idx):
        px, qx, _ = self.prepare_batch(batch)
        loss_dict = self.train_step_fn(self.model, (qx, px), step=self.global_step)
        self.log("train_loss", loss_dict["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=px.size(0))
        return loss_dict["loss"]

    def compute_cpd_statistic(self, Y):
        Y = Y.float()
        T, d = Y.shape
        Y_proc = self.preprocess(Y)
        log_ratio, _ = self.density_ratio_fn(self.model, Y_proc, joint=None)
        score = log_ratio
        w = self.swl // 2
        cp_stat = torch.zeros(T, device=Y.device)
        for t in range(w, T - w):
            left_mean = score[t - w:t].mean()
            right_mean = score[t:t + w].mean()
            cp_stat[t] = torch.abs(left_mean - right_mean)
        return cp_stat

    def on_epoch_end(self, step_outputs):
        if not self.trainer.is_global_zero:
            return
        cp_stats = [out["cp_stat"].cpu().numpy() for out in step_outputs]
        if not cp_stats:
            return
        true_cps = None
        if "text" in self.args.data:
            true_cps = [1014, 1608]
        elif "creditcard" in self.args.data:
            true_cps = [1000, 1492]
        vis_utils.visualize_cpd_stat(cp_stats=cp_stats, window_length=self.swl, true_change_points=true_cps, save_path=os.path.join(self.metrics_dir, "cpd_stat_summary.pdf"), title="Change Point Detection: %s" % self.args.data)

    def validation_step(self, batch, batch_idx):
        Y = batch[0] if isinstance(batch, (list, tuple)) else batch
        if Y.dim() == 3:
            Y = Y.squeeze(0)
        cp_stat = self.compute_cpd_statistic(Y)
        save_path = os.path.join(self.metrics_dir, "cpd_stat_val_batch_%s.npy" % batch_idx)
        np.save(save_path, cp_stat.cpu().numpy())
        self.validation_step_outputs.append({"cp_stat": cp_stat})

    def on_validation_epoch_end(self):
        self.on_epoch_end(self.validation_step_outputs)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        Y = batch[0] if isinstance(batch, (list, tuple)) else batch
        if Y.dim() == 3:
            Y = Y.squeeze(0)
        cp_stat = self.compute_cpd_statistic(Y)
        save_path = os.path.join(self.metrics_dir, "cpd_stat_test_batch_%s.npy" % batch_idx)
        np.save(save_path, cp_stat.cpu().numpy())
        self.test_step_outputs.append({"cp_stat": cp_stat})

    def on_test_epoch_end(self):
        self.on_epoch_end(self.test_step_outputs)
        self.test_step_outputs.clear()
