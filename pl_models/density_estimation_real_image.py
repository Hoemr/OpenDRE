# -*- coding: utf-8 -*-
"""Density estimation: real image."""
import os
import time
import math
import pickle

import numpy as np
import torch
import torch.optim as optim
import torchvision.utils as vutils

import nn_models as base_models
import dre.losses.losses as train_step_fns
from pl_data import DATA_SHAPES, SPECS
from dre.path import ImageInterpXt
import dre_utils.visualization as vis_utils
from dre_utils.sampling import sampling

from .base import DensityEstimationModel


class DensityEstimationRealImageModel(DensityEstimationModel):
    def __init__(self, args, save_path):
        super().__init__(args, save_path)
        self.save_hyperparameters()
        self.subtask = args.subtask
        self.noise_level = 0.
        self.data_type = args.data
        self.data_shape = DATA_SHAPES[self.data_type]

    def _build_sde(self, args):
        return ImageInterpXt(args=args)

    def set_basic_functions(self, args):
        self.path = self._build_sde(args)
        self.likelihood_fn = self.get_density_fn()
        self.density_ratio_fn = self.get_dr_fn()
        self.train_step_fn = train_step_fns.ScoreMatchingTrainStepFn(args, self.path)
        self.model = self.build_model()
        self.train_epoch_end_step_fn = lambda: None
        use_pretrained_flow = getattr(args, "use_pretrained_flow", False)
        if use_pretrained_flow:
            # Use NSF flow as a fixed data transform (requires checkpoints and data stats).
            self.flow = self.load_pretrained_flow_model()
            self.flow.eval()
            for param in self.flow.parameters():
                param.requires_grad = False

            def flow_fn(batch, train=False):
                shape = batch.shape
                with torch.no_grad():
                    x_processed = batch * 255.0 / 256.0
                    x_processed += torch.rand_like(x_processed) / 256.0
                    x_processed *= 256.0
                    batch_z, log_det_J = self.flow.transform_to_noise(
                        x_processed, logdet=True, transform=True, train=train
                    )
                    batch_z = batch_z.reshape(shape)
                return batch_z, log_det_J

            self.flow_fn = flow_fn
        else:
            # Default: identity transform, no NSF dependency.
            def flow_fn(batch, train=False):
                return batch, None

            self.flow_fn = flow_fn

    def load_pretrained_flow_model(self):
        from nsf.experiments.images_noise import create_transform
        from nsf.nde import distributions, transforms, flows

        with open(os.path.join("nsf", "flow_ckpts", "test_data_stats.p"), "rb") as fp:
            data_stats = pickle.load(fp)
        device = self.args.device
        train_mean = data_stats["train_mean"].to(device)
        test_mean = data_stats["test_mean"].to(device)
        val_mean = data_stats["val_mean"].to(device)
        train_cov = data_stats["train_cov_cholesky"].to(device)
        val_cov = data_stats["val_cov_cholesky"].to(device)
        c, h, w = 1, 28, 28
        spline_params = {
            "apply_unconditional_transform": False,
            "min_bin_height": 0.001,
            "min_bin_width": 0.001,
            "min_derivative": 0.001,
            "num_bins": 128,
            "tail_bound": 3.0,
        }
        distribution = distributions.StandardNormal((c * h * w,))
        train_transform, val_transform, transform = create_transform(
            c,
            h,
            w,
            train_mean,
            val_mean,
            train_cov,
            val_cov,
            levels=2,
            hidden_channels=64,
            steps_per_level=8,
            alpha=0.000001,
            num_bits=8,
            preprocessing="realnvp_2alpha",
            multi_scale=False,
            actnorm=True,
            coupling_layer_type="rational_quadratic_spline",
            spline_params=spline_params,
            use_resnet=False,
            num_res_blocks=2,
            resnet_batchnorm=False,
            dropout_prob=0.0,
        )
        net = flows.FlowDataTransform(transform, distribution, train_transform, val_transform)
        net = net.to(device)
        return net

    def build_model(self):
        hidden_channels = list(map(int, self.args.dims.split("-")))
        in_channels = self.args.in_channels
        image_backbone = getattr(self.args, "image_backbone", "resnet18")
        image_dropout = getattr(self.args, "image_dropout", 0.3)

        if "unet" == image_backbone: 
            score_model = base_models.ImageJointScoreModel
        else:
            score_model = base_models.ImageJointScoreModelResNet
            
        return score_model(
            args=self.args,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            embed_dim=self.args.embed_dim,
            nonlinearity=self.args.nonlinearity,
            backbone=image_backbone,
            dropout=image_dropout
        )

    def get_bpd_by_nll(self, nll, x=None, log_det_J=None):
        if x is not None:
            dim = np.prod(x.shape[1:])
        else:
            dim = np.prod(DATA_SHAPES[self.data_type])
        if log_det_J is not None:
            # Flow path: convert from latent-space NLL to data-space NLL using log|det J|.
            if hasattr(log_det_J, "mean"):
                nll = nll - log_det_J.mean()
            else:
                nll = nll - float(log_det_J)
        else:
            # Identity / normalized-data path: account for x -> (x-mean)/std normalization.
            channels, _, stds = SPECS[self.data_type]
            log_sigma_per_channel = sum(math.log(s) for s in stds)
            pixels_per_channel = dim / channels
            log_det_correction = pixels_per_channel * log_sigma_per_channel
            nll += log_det_correction

        # Report BPD in standard 8-bit units for readability/comparison.
        # bpd8 = (-log p(x) + dim * log 256) / (dim * log 2)
        bpd_8bit = (nll + dim * math.log(256.0)) / (math.log(2.0) * dim)
        return nll, bpd_8bit

    def sampling(self, n_samples, method="HMC", num_steps=100, step_size=0.1, filename="best", **kwargs):
        with torch.enable_grad():
            z_samples = sampling(
                self.model, self.path, self.density_ratio_fn,
                n_samples=n_samples, data_shape=self.data_shape,
                joint=self.joint, method=method,
                num_steps=num_steps, step_size=step_size,
                device=self.device, normalize_const=self.normalize_const,
                save_dir=self.sample_path, filename=filename, **kwargs
            )
        with torch.no_grad():
            flow_model = self.flow.module if hasattr(self.flow, "module") else self.flow
            samples_x = flow_model.sample(
                z_samples.view(n_samples, -1),
                context=None, rescale=True, transform=True, train=False
            )
            samples_x = samples_x.view(n_samples, *self.data_shape)
        samples = (samples_x + 1.) / 2.
        samples = torch.clamp(samples, 0, 1)
        nrow = min(16, int(math.sqrt(n_samples)))
        grid = vutils.make_grid(samples, nrow=nrow, padding=2, normalize=False)
        vutils.save_image(grid, os.path.join(self.sample_path, "samples_best.png"))

    def stage_log(self, stage_name, stage_loss=0., stage_nll=0., batch_size=64, nfe=1., bpd=0., time=0.):
        if hasattr(stage_loss, "item"):
            stage_loss = stage_loss.to(torch.float32)
        stage_nll = stage_nll.to(torch.float32)
        self.log("%s_bpd" % stage_name, bpd, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        super().stage_log(stage_name=stage_name, stage_loss=stage_loss, stage_nll=stage_nll,
                         batch_size=batch_size, nfe=nfe, time=time)

    def training_step(self, batch, batch_idx):
        batch = batch.to(self.device)
        shape = batch.shape
        noise = self.path.prior_sampling(shape).to(batch)
        batch, log_det_J = self.flow_fn(batch, train=True)
        loss_dict = self.train_step_fn(self.model, (noise, batch), step=self.global_step, log_det_J=log_det_J)
        nll, bpd = self.get_bpd_by_nll(loss_dict["nll"], batch, log_det_J)
        self.stage_log("train", loss_dict["loss"], nll, batch.size(0), bpd=bpd)
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        batch, log_det_J = self.flow_fn(batch, train=False)
        start_time = time.time()
        val_logp, val_nfe = self.likelihood_fn(self.model, batch, joint=False)
        end_time = time.time()
        val_time = end_time - start_time
        val_nll = -val_logp.mean()
        nll, bpd = self.get_bpd_by_nll(val_nll, batch, log_det_J)
        self.stage_log("val", stage_nll=nll, batch_size=batch.size(0), nfe=val_nfe, bpd=bpd, time=val_time)
        if batch_idx == 0:
            vis_utils.plot_integral_distributions(
                self.model, batch, self.sample_path,
                is_integral=self.args.SAI,
                path_type=self.args.path_type,
                epoch="final"
            )
        return {"NLL": nll, "nfe": val_nfe}

    def on_validation_epoch_end(self, outputs=None):
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
        batch, log_det_J = self.flow_fn(batch, train=False)
        start_time = time.time()
        test_logp, test_nfe = self.likelihood_fn(self.model, batch, joint=False)
        end_time = time.time()
        test_time = end_time - start_time
        test_nll = -test_logp.mean()
        nll, bpd = self.get_bpd_by_nll(test_nll, batch, log_det_J)
        self.stage_log("test", stage_nll=nll, batch_size=batch.size(0), nfe=test_nfe, bpd=bpd, time=test_time)
        return {"NLL": nll, "nfe": test_nfe}

    def configure_optimizers(self):
        if self.args.sub_method == "kernel":
            self.automatic_optimization = False
            return []
        if self.args.path_type == "mvp":
            optimizer = optim.AdamW([
                {"params": self.model.parameters(), "lr": self.args.lr, "weight_decay": 1e-5},
                {"params": self.train_step_fn.parameters(), "lr": self.args.lr, "weight_decay": 0},
            ])
        else:
            optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
            }
        }
