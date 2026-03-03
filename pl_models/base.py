# -*- coding: utf-8 -*-
"""Base Lightning modules: DensityEstimationModel and DensityRatioEstimationModel."""
import os
import time
import pickle

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
import torchvision.utils as vutils
from typing import Tuple

from path import ToyInterpXt, ImageInterpXt
import nn_models as base_models
from estimators import get_likelihood_fn, get_density_ratio_fn, MetricEvaluator
import losses.losses as train_step_fns
from pl_data import SPECS, DATA_SHAPES
import visualization.visualization as vis_utils
from sampling import sampling


class DensityEstimationModel(pl.LightningModule):
    def __init__(self, args, save_path):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.task = args.task
        self.sub_method = getattr(args, "sub_method", "score")
        self.subsub_method = getattr(args, "subsub_method", "dre_infty")
        self.joint = args.joint
        self.decomp = args.decomp
        self.eps = args.eps
        self.noise_level = 0.
        self.set_basic_functions(args)

        self.likelihood_path = os.path.join(save_path, "likelihood")
        self.checkpoint_dir = os.path.join(save_path, "checkpoints")
        self.sample_path = os.path.join(save_path, "samples")
        os.makedirs(self.likelihood_path, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)

        self.data_type = args.data
        self.normalize_const = SPECS[self.data_type] if self.data_type in SPECS else 1.

    def _build_sde(self, args):
        return ToyInterpXt(args=args)

    def set_basic_functions(self, args):
        self.path = self._build_sde(args)
        self.likelihood_fn = self.get_density_fn()
        self.density_ratio_fn = self.get_dr_fn()
        self.model = self.build_model()
        self.train_epoch_end_step_fn = lambda: None

        if self.args.sub_method == "kernel":
            self.train_step_fn = train_step_fns.KernelDensityRatioTrainStepFn(
                args, path_model=self.path, method=self.subsub_method
            )
            self.train_epoch_end_step_fn = self.model.train_model
        elif self.args.sub_method == "neural":
            self.train_step_fn = train_step_fns.NeuralDensityRatioTrainStepFn(
                args, path_model=self.path, method=self.subsub_method
            )
        elif self.args.sub_method == "score":
            self.train_step_fn = train_step_fns.ScoreMatchingTrainStepFn(args, self.path)
        else:
            raise ValueError("Invalid sub_method %s" % self.args.sub_method)

    def build_model(self):
        hidden_dims = list(map(int, self.args.dims.split("-")))
        if self.args.sub_method == "kernel":
            return base_models.KernelDensityRatioModel(
                self.args, path_model=self.path, method=self.subsub_method
            )
        elif self.args.sub_method == "neural":
            return base_models.NeuralDensityRatioModel(
                input_dim=self.args.d,
                hidden_dims=hidden_dims,
                nonlinearity=self.args.nonlinearity,
                args=self.args,
                norm=self.args.norm,
                use_spectral_norm=self.args.specnorm,
            )
        else:
            if self.decomp:
                return base_models.DecompJointScoreModel(
                    input_dim=self.args.d,
                    hidden_dims=hidden_dims,
                    embed_dim=self.args.embed_dim,
                    nonlinearity=self.args.nonlinearity,
                    args=self.args,
                    norm=self.args.norm,
                    use_spectral_norm=self.args.specnorm,
                    K=self.args.K_decomp,
                    polynomials=self.args.polynomials,
                    noise_type=self.args.noise_type,
                )
            else:
                return base_models.JointScoreModel(
                    input_dim=self.args.d,
                    hidden_dims=hidden_dims,
                    embed_dim=self.args.embed_dim,
                    nonlinearity=self.args.nonlinearity,
                    args=self.args,
                    norm=self.args.norm,
                    use_spectral_norm=self.args.specnorm,
                )

    def get_density_fn(self):
        return get_likelihood_fn(self.path, args=self.args, quad_step=self.args.quad_step)

    def get_dr_fn(self):
        return get_density_ratio_fn(self.path, args=self.args, quad_step=self.args.quad_step)

    def sampling(self, n_samples, method="HMC", num_steps=100, step_size=0.1, filename="best", **kwargs):
        with torch.enable_grad():
            samples = sampling(
                self.model, self.path, self.density_ratio_fn,
                n_samples=n_samples, data_shape=self.data_shape,
                joint=self.joint, method=method,
                num_steps=num_steps, step_size=step_size,
                device=self.device, normalize_const=self.normalize_const,
                save_dir=self.sample_path, filename=filename, **kwargs
            )
        grid = vutils.make_grid(samples, nrow=16, padding=2, normalize=False)
        vutils.save_image(grid, os.path.join(self.sample_path, "samples_best.png"))

    def forward(self, t, x):
        return self.model(t, x)

    def stage_log(self, stage_name, stage_loss=0, stage_nll=0, batch_size=64, nfe=1., time=0.):
        self.log("%s_loss" % stage_name, stage_loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        self.log("%s_nll" % stage_name, stage_nll, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        self.log("%s_nfe" % stage_name, float(nfe), on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        self.log("%s_time" % stage_name, float(time), on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True, reduce_fx="sum")

    def training_step(self, batch, batch_idx):
        noise = torch.randn_like(batch) * self.noise_level if self.noise_level > 0 else 0.
        batch = batch + noise
        noise = self.path.prior_sampling(batch.shape).to(batch)
        loss_dict = self.train_step_fn(self.model, (noise, batch), step=self.global_step)
        self.stage_log("train", loss_dict["loss"], loss_dict["nll"], batch.size(0))
        return loss_dict["loss"]

    def on_train_epoch_end(self):
        self.train_epoch_end_step_fn()
        super().on_train_epoch_end()

    def configure_optimizers(self):
        if self.args.sub_method == "kernel":
            self.automatic_optimization = False
            return []
        if self.args.combination_type == "gpp":
            optimizer = optim.AdamW([
                {"params": self.model.parameters(), "lr": self.args.lr, "weight_decay": 1e-5},
                {"params": self.train_step_fn.parameters(), "lr": self.args.lr, "weight_decay": 0},
            ])
        else:
            optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=1e-5)
        return {"optimizer": optimizer}

    def on_load_checkpoint(self, checkpoint):
        self.model.load_state_dict(checkpoint["state_dict"], strict=False)


class DensityRatioEstimationModel(pl.LightningModule):
    def __init__(self, args, save_path, datamodule):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.task = args.task
        self.subtask = args.subtask
        self.sub_method = getattr(args, "sub_method", "score")
        self.subsub_method = getattr(args, "subsub_method", "dre_infty")
        self.data_type = args.data
        self.d = args.d
        self.decomp = args.decomp
        self.joint = args.joint
        self.eps = args.eps
        self.set_basic_functions(args)
        self.set_prepare_batch()
        self.datamodule = datamodule

        self.fdiv_path = os.path.join(save_path, "fdiv")
        self.checkpoint_dir = os.path.join(save_path, "checkpoints")
        self.metrics_dir = os.path.join(save_path, "metrics")
        self.sample_path = os.path.join(save_path, "samples")
        os.makedirs(self.fdiv_path, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)

        self.metric_evaluator = MetricEvaluator()
        self.metrics_db = []
        self.fdiv_db = []
        self.fdiv_std_db = []
        self.fdiv_true_db = []
        self.mae_db = []
        self.nfe_db = []
        self.epoch_db = []
        self.validation_step_outputs = []

    def _build_sde(self, args):
        return ToyInterpXt(args=args)

    def set_prepare_batch(self):
        if self.subtask == "mi_realdata":
            if self.data_type == "image":
                def prepare_batch(batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                    x, y, true_mi = batch
                    px = torch.cat([x, y], dim=1)
                    batch_size = x.size(0)
                    idx = torch.randperm(batch_size)
                    y_shuffled = y[idx]
                    qx = torch.cat([x, y_shuffled], dim=1)
                    return px, qx, true_mi
            else:
                def prepare_batch(batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                    x, y, true_mi = batch
                    px = torch.cat([x, y], dim=1)
                    idx = torch.randperm(x.size(0))
                    y_shuffled = y[idx]
                    qx = torch.cat([x, y_shuffled], dim=1)
                    return px, qx, true_mi
        else:
            def prepare_batch(batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                px, qx, true_mi = batch
                if self.subtask == "ood_detection" and self.training:
                    noise_std = getattr(self.args, "sample_noise_std", 0.0)
                    if noise_std > 0:
                        px = px + torch.randn_like(px) * noise_std
                        qx = qx + torch.randn_like(qx) * noise_std
                return px, qx, true_mi
        self.prepare_batch = prepare_batch

    def set_basic_functions(self, args):
        self.path = self._build_sde(args)
        self.density_ratio_fn = self.get_dr_fn()
        self.model = self.build_model()
        self.train_epoch_end_step_fn = lambda: None

        if self.args.sub_method == "kernel":
            self.train_step_fn = train_step_fns.KernelDensityRatioTrainStepFn(
                args, path_model=self.path, method=self.subsub_method
            )
            self.train_epoch_end_step_fn = self.model.train_model
        elif self.args.sub_method == "neural":
            self.train_step_fn = train_step_fns.NeuralDensityRatioTrainStepFn(
                args, path_model=self.path, method=self.subsub_method
            )
        elif self.args.sub_method == "score":
            self.train_step_fn = train_step_fns.ScoreMatchingTrainStepFn(args, self.path)
        else:
            raise ValueError("Invalid sub_method %s" % self.args.sub_method)

    def _build_model_with_image_backbone(self):
        image_backbone = getattr(self.args, "image_backbone", "resnet18")
        image_dropout = getattr(self.args, "image_dropout", 0.3)
        hidden_channels = list(map(int, self.args.dims.split("-")))
        in_channels = 3
        if self.args.sub_method == "neural":
            return base_models.ResNetDensityRatioModel(
                in_channels=in_channels,
                num_classes=1,
                backbone=image_backbone,
                dropout=image_dropout
            )
        elif self.args.sub_method == "score":
            if self.decomp:
                return base_models.ImageDecompJointScoreModelResNet(
                    args=self.args,
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    embed_dim=self.args.embed_dim,
                    K=self.args.K_decomp,
                    polynomials=self.args.polynomials,
                    noise_type=self.args.noise_type,
                    backbone=image_backbone,
                    dropout=image_dropout
                )
            else:
                return base_models.ImageJointScoreModel(
                    args=self.args,
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    embed_dim=self.args.embed_dim,
                    backbone=image_backbone,
                    dropout=image_dropout
                )
        else:
            hidden_dims = list(map(int, self.args.dims.split("-")))
            return base_models.KernelDensityRatioModel(
                self.args, path_model=self.path, method=self.subsub_method
            )

    def _build_default_model(self):
        hidden_dims = list(map(int, self.args.dims.split("-")))
        if self.args.sub_method == "kernel":
            return base_models.KernelDensityRatioModel(
                self.args, path_model=self.path, method=self.subsub_method
            )
        elif self.args.sub_method == "neural":
            return base_models.NeuralDensityRatioModel(
                input_dim=self.args.d,
                hidden_dims=hidden_dims,
                nonlinearity=self.args.nonlinearity,
                args=self.args,
                norm=self.args.norm,
                use_spectral_norm=self.args.specnorm,
            )
        else:
            if self.decomp:
                return base_models.DecompJointScoreModel(
                    input_dim=self.d,
                    hidden_dims=hidden_dims,
                    embed_dim=self.args.embed_dim,
                    K=self.args.K_decomp,
                    polynomials=self.args.polynomials,
                    nonlinearity=self.args.nonlinearity,
                    noise_type=self.args.noise_type,
                    args=self.args,
                    norm=self.args.norm,
                    use_spectral_norm=self.args.specnorm,
                )
            else:
                return base_models.JointScoreModel(
                    input_dim=self.d,
                    hidden_dims=hidden_dims,
                    embed_dim=self.args.embed_dim,
                    nonlinearity=self.args.nonlinearity,
                    args=self.args,
                    norm=self.args.norm,
                    use_spectral_norm=self.args.specnorm,
                )

    def build_model(self):
        if self.args.data in DATA_SHAPES.keys():
            return self._build_model_with_image_backbone()
        return self._build_default_model()

    def get_dr_fn(self):
        return get_density_ratio_fn(self.path, args=self.args, quad_step=self.args.quad_step)

    def forward(self, t, x):
        return self.model(t, x)

    def training_step(self, batch, batch_idx):
        px, qx, _ = self.prepare_batch(batch)
        loss_dict = self.train_step_fn(self.model, (qx, px), step=self.global_step)
        self.log("train_loss", loss_dict["loss"], on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True, batch_size=px.size(0))
        return loss_dict["loss"]

    def on_train_epoch_end(self):
        self.train_epoch_end_step_fn()
        super().on_train_epoch_end()

    def validation_step(self, batch, batch_idx):
        px, _, true_fdiv = self.prepare_batch(batch)
        true_fdiv = true_fdiv.mean().cpu().item()
        log_pq, est_nfe = self.density_ratio_fn(self.model, px, joint=self.joint)
        est_fdiv = log_pq.squeeze().mean()
        mae = torch.abs(est_fdiv - true_fdiv).item()
        metrics = self.metric_evaluator.evaluate(true_fdiv, log_pq)
        self.log_dict({
            "val_true_fdiv": true_fdiv,
            "val_est_fdiv": est_fdiv,
            "val_mae": mae,
            "val_nfe": float(est_nfe)
        }, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=px.shape[0])
        step_output = {
            "true_fdiv": true_fdiv,
            "est_fdiv": est_fdiv.cpu(),
            "est_fdiv_std": log_pq.std().cpu(),
            "est_nfe": est_nfe,
            "mae": mae,
            "metrics": metrics,
            "batch_size": px.shape[0],
        }
        self.validation_step_outputs.append(step_output)
        if batch_idx == 0:
            vis_utils.plot_integral_distributions(
                self.model, px, self.sample_path,
                is_integral=self.args.SAI,
                combination_type=self.args.combination_type,
                epoch="final"
            )
        return step_output

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        total_samples = sum(out["batch_size"] for out in outputs)
        weighted_keys = ["true_fdiv", "est_fdiv", "est_fdiv_std", "est_nfe", "mae"]
        summed_metrics = {}
        for key in weighted_keys:
            summed_metrics[key] = sum(out[key] * out["batch_size"] for out in outputs) / total_samples
        true_fdiv = summed_metrics["true_fdiv"]
        all_metrics = [out["metrics"] for out in outputs]
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        self.metrics_db.append(avg_metrics)
        self.fdiv_db.append(summed_metrics["est_fdiv"])
        self.fdiv_std_db.append(summed_metrics["est_fdiv_std"])
        self.fdiv_true_db.append(true_fdiv)
        self.mae_db.append(summed_metrics["mae"])
        self.nfe_db.append(summed_metrics["est_nfe"])
        self.epoch_db.append(self.current_epoch)
        self.validation_step_outputs.clear()
        vis_utils.visualize_fdiv(self.args, self.fdiv_db, self.fdiv_true_db, savefig=self.fdiv_path)
        vis_utils.visualize_path_function(
            self.path, sample_path=self.sample_path,
            combination_type=self.args.combination_type,
            constraint_type=self.args.constraint_type
        )
        vis_utils.visualize_variance_profiles(
            self.path, sample_path=self.sample_path,
            combination_type=self.args.combination_type
        )
        with open(os.path.join(self.metrics_dir, "metrics.p"), "wb") as f:
            pickle.dump({
                "true_fdiv": self.fdiv_true_db,
                "est_fdiv": self.fdiv_db,
                "mae": self.mae_db,
                "nfe": self.nfe_db,
                "epoch": self.epoch_db,
                "metrics": self.metrics_db,
            }, f)

    def test_step(self, batch, batch_idx):
        px, _, true_fdiv = self.prepare_batch(batch)
        true_fdiv = true_fdiv.mean().cpu().item()
        start_time = time.time()
        log_pq, est_nfe = self.density_ratio_fn(self.model, px, joint=self.joint)
        end_time = time.time()
        test_time = end_time - start_time
        est_fdiv = log_pq.squeeze().mean()
        mae = torch.abs(est_fdiv - true_fdiv).item()
        metrics = self.metric_evaluator.evaluate(true_fdiv, log_pq)
        all_metrics = {
            "true_fdiv": true_fdiv,
            "est_fdiv": est_fdiv,
            "nfe": est_nfe,
            "MAE": mae,
            **metrics
        }
        if batch_idx == 0:
            vis_utils.plot_integral_distributions(
                self.model, px, self.sample_path,
                is_integral=self.args.SAI,
                combination_type=self.args.combination_type,
                epoch="final"
            )
        vis_utils.visualize_variance_profiles(
            self.path, sample_path=self.sample_path,
            combination_type=self.args.combination_type
        )
        vis_utils.visualize_integrated_variance_profiles(
            self.path, sample_path=self.sample_path,
            combination_type=self.args.combination_type
        )
        for key, value in all_metrics.items():
            if isinstance(value, (int, float)):
                self.log(key, float(value), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            else:
                self.log(key, value, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("test_time", float(test_time), on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True, reduce_fx="sum")

    def configure_optimizers(self):
        if self.args.sub_method == "kernel":
            self.automatic_optimization = False
            return []
        if self.args.combination_type == "gpp":
            optimizer = optim.AdamW([
                {"params": self.model.parameters(), "lr": self.args.lr, "weight_decay": 1e-5},
                {"params": self.train_step_fn.parameters(), "lr": self.args.lr, "weight_decay": 0},
            ])
        else:
            optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=1e-5)
        return optimizer
