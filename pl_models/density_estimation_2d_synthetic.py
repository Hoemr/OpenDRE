# -*- coding: utf-8 -*-
"""Density estimation: 2D synthetic."""
import time
import utils.visualization as vis_utils
from .base import DensityEstimationModel

class DensityEstimation2DSyntheticModel(DensityEstimationModel):
    def __init__(self, args, save_path):
        super().__init__(args, save_path)
        self.subtask = "synthetic2d"
        self.noise_level = 0.
        self.set_path_plot_fn()

    def set_path_plot_fn(self):
        if self.args.sub_method in ["neural", "kernel"]:
            def path_plot_fn(batch, epoch="final"):
                pass
        elif self.args.sub_method == "score":
            def path_plot_fn(batch, epoch="final"):
                vis_utils.visualize_path_function(self.path, sample_path=self.sample_path, path_type=self.args.path_type, constraint_type=self.args.constraint_type)
                vis_utils.plot_integral_distributions(self.model, batch, self.sample_path, is_integral=self.args.SAI, path_type=self.args.path_type, epoch=epoch)
        else:
            raise NotImplementedError("Submethod not implemented")
        self.path_plot_fn = path_plot_fn

    def validation_step(self, batch, batch_idx):
        start_time = time.time()
        NFE = vis_utils.plt_likelihood_dre(self.args, self.model, self.likelihood_fn, savefig=self.likelihood_path, epoch=self.current_epoch+1, npts=800, device=self.device, steps=self.args.quad_step)
        end_time = time.time()
        self.path_plot_fn(batch, epoch="final")
        self.log("val_nfe", NFE, prog_bar=True)
        self.log("val_time", end_time - start_time, prog_bar=True)

    def test_step(self, batch, batch_idx):
        epoch = "solver_%s_steps_%s" % (self.args.solver, self.args.quad_step)
        start_time = time.time()
        NFE = vis_utils.plt_likelihood_dre(self.args, self.model, self.likelihood_fn, savefig=self.likelihood_path, epoch=epoch, npts=800, device=self.device, steps=self.args.quad_step)
        _ = vis_utils.plt_density_ratio(self.args, self.model, self.density_ratio_fn, savefig=self.likelihood_path, epoch=epoch, npts=800, device=self.device, steps=self.args.quad_step)
        vis_utils.plt_standard_normal_likelihood(self.args, npts=800, savefig=self.likelihood_path, epoch=epoch, device=self.device)
        end_time = time.time()
        self.path_plot_fn(batch, epoch="final")
        self.log("test_nfe", NFE, prog_bar=True)
        self.log("test_time", end_time - start_time, prog_bar=True)
