# -*- coding: utf-8 -*-
"""Density estimation: tabular."""
import time
import utils.visualization as vis_utils
from .base import DensityEstimationModel

class DensityEstimationTabularModel(DensityEstimationModel):
    def __init__(self, args, save_path):
        super().__init__(args, save_path)
        self.subtask = "tabular"

    def validation_step(self, batch, batch_idx):
        start_time = time.time()
        val_logp, val_nfe = self.likelihood_fn(self.model, batch, joint=self.joint)
        end_time = time.time()
        val_nll = -val_logp.mean()
        self.stage_log("val", stage_nll=val_nll, batch_size=batch.size(0), nfe=val_nfe, time=end_time - start_time)
        return {"NLL": val_nll, "nfe": val_nfe}

    def on_validation_epoch_end(self, outputs=None):
        vis_utils.visualize_path_function(self.path, sample_path=self.sample_path, path_type=self.args.path_type, constraint_type=self.args.constraint_type)

    def test_step(self, batch, batch_idx):
        start_time = time.time()
        test_logp, test_nfe = self.likelihood_fn(self.model, batch, joint=self.joint, steps=self.args.quad_step)
        end_time = time.time()
        test_nll = -test_logp.mean()
        self.stage_log("test", stage_nll=test_nll, batch_size=batch.size(0), nfe=test_nfe, time=end_time - start_time)
        return {"NLL": test_nll, "nfe": test_nfe}
