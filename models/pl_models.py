# -*- coding: utf-8 -*-
import os
import time
import math
import pickle
import numpy as np
from typing import Tuple, Optional
import pytorch_lightning as pl
import torch
import torch.optim as optim
from torchvision.utils import save_image
import torchvision.utils as vutils
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F

from models.SDE import ToyInterpXt, ImageInterpXt
import models.base_models as base_models
from models.estimators import get_likelihood_fn, get_density_ratio_fn, MetricEvaluator
import models.losses as train_step_fns

from .pl_data import SPECS, DATA_SHAPES
import utils.visualization as vis_utils
from utils.sampling import sampling


class DensityEstimationModel(pl.LightningModule):
    def __init__(self, args, save_path):
        super().__init__()
        self.save_hyperparameters()
        # Common attributes from args
        self.args = args
        self.task = args.task
        self.sub_method = getattr(args, "sub_method", "score")
        self.subsub_method = getattr(args, "subsub_method", "dre_infty")
        
        self.joint = args.joint
        self.decomp = args.decomp
        self.eps = args.eps
        self.noise_level = 0.
        
        self.set_basic_functions(args)
        
        # Paths for saving outputs
        self.likelihood_path = os.path.join(save_path, "likelihood")
        self.checkpoint_dir = os.path.join(save_path, "checkpoints")
        self.sample_path = os.path.join(save_path, "samples")
        os.makedirs(self.likelihood_path, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)
        
        # for sampling
        self.data_type = args.data
        self.normalize_const=SPECS[self.data_type] if self.data_type in SPECS else 1.
        
    def set_basic_functions(self, args):
        self.sde = ToyInterpXt(args=args)
        self.likelihood_fn = self.get_density_fn()
        self.density_ratio_fn = self.get_dr_fn()
        self.model = self.build_model()
        self.train_epoch_end_step_fn = lambda : None
        
        if self.args.sub_method == "kernel":
            self.train_step_fn = train_step_fns.KernelDensityRatioTrainStepFn(
                args, sde=self.sde, method=self.subsub_method
            )
            self.train_epoch_end_step_fn = self.model.train_model
            
        elif self.args.sub_method == "neural":
            self.train_step_fn = train_step_fns.NeuralDensityRatioTrainStepFn(
                args, sde=self.sde, method=self.subsub_method
            )
            
        elif self.args.sub_method == "score":
            self.train_step_fn = train_step_fns.ScoreMatchingTrainStepFn(args, self.sde)
            
        else:
            raise ValueError(f"Invalid sub_method {self.args.sub_method}")
        
    def build_model(self):
        """Constructs the neural network based on decomposition choice."""
        hidden_dims = list(map(int, self.args.dims.split("-")))
        if self.args.sub_method == "kernel":
            return base_models.KernelDensityRatioModel(
                self.args, sde=self.sde, method=self.subsub_method
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
        """Returns the appropriate likelihood function based on decomposition."""
        return get_likelihood_fn(self.sde, args=self.args, quad_step=self.args.quad_step)

    def get_dr_fn(self):
        """return density ratio estimation functions"""
        return get_density_ratio_fn(self.sde, args=self.args, quad_step=self.args.quad_step)
    
    def sampling(self, n_samples, method="HMC", num_steps=100, step_size=0.1, filename="best", **kwargs):
        with torch.enable_grad():
            samples = sampling(self.model, self.sde, self.density_ratio_fn, 
                            n_samples=n_samples, data_shape=self.data_shape, 
                            joint=self.joint, method=method, 
                            num_steps=num_steps, step_size=step_size,
                            device=self.device, normalize_const=self.normalize_const,
                            save_dir=self.sample_path, filename=filename, **kwargs)
        # # normalize
        # channels, mean, std = self.normalize_const
        # device = samples.device
        # mean = torch.tensor(mean, device=device).view(1, channels, 1, 1)
        # std = torch.tensor(std, device=device).view(1, channels, 1, 1)
        # samples = samples * std + mean
        # samples = samples * 255.0         # 缩放到 [0, 255]
        # samples = torch.clamp(samples, 0, 255).to(torch.uint8)
        
        grid = vutils.make_grid(samples, nrow=16, padding=2, normalize=False)
        vutils.save_image(grid, os.path.join(self.sample_path, "samples_best.png"))
        # save_image(samples, os.path.join(self.sample_path, "samples_best.png"), nrow=2)
        # return samples

    def forward(self, t, x):
        return self.model(t, x)
    
    def stage_log(self, stage_name, stage_loss=0, stage_nll=0, batch_size=64, nfe=1., time=0.):
        self.log(f"{stage_name}_loss", stage_loss, on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)       
        self.log(f"{stage_name}_nll", stage_nll, on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        self.log(f"{stage_name}_nfe", float(nfe), on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{stage_name}_time", float(time), on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True, reduce_fx="sum")

    def training_step(self, batch, batch_idx):
        noise = torch.randn_like(batch) * self.noise_level if self.noise_level > 0 else 0.
        batch = batch + noise
        noise = self.sde.prior_sampling(batch.shape).to(batch)

        loss_dict = self.train_step_fn(self.model, (noise, batch), step=self.global_step) 
        self.stage_log("train", loss_dict["loss"], loss_dict["nll"], batch.size(0))
        return loss_dict["loss"]
    
    def on_train_epoch_end(self):
        self.train_epoch_end_step_fn()
        super().on_train_epoch_end()

    def configure_optimizers(self):
        if self.args.sub_method == "kernel":
            self.automatic_optimization = False    # do not need to call optimizer.step() under kernel method
            return []
        
        if self.args.combination_type == "gpp":
            optimizer = optim.AdamW([
                {'params': self.model.parameters(), 'lr': self.args.lr, 'weight_decay': 1e-5},
                {'params': self.train_step_fn.parameters(), 'lr': self.args.lr, 'weight_decay': 0},
            ])
        else:
            optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=1e-5)
        
        return {"optimizer": optimizer}
    
    def on_load_checkpoint(self, checkpoint):
        self.model.load_state_dict(checkpoint["state_dict"], strict=False)

class DensityEstimation2DSyntheticModel(DensityEstimationModel):
    def __init__(self, args, save_path):
        super().__init__(args, save_path)
        self.subtask = "synthetic2d"
        self.noise_level = 0. #args.GRBC_epsilon
        self.set_path_plot_fn()
        
    def set_path_plot_fn(self):
        if self.args.sub_method in ["neural", "kernel"]:
            def path_plot_fn(batch, epoch="final"):
                pass
        elif self.args.sub_method == "score":
            def path_plot_fn(batch, epoch="final"):
                vis_utils.visualize_path_function(self.sde, sample_path=self.sample_path, combination_type=self.args.combination_type, constraint_type=self.args.constraint_type)
                vis_utils.plot_integral_distributions(self.model, batch, self.sample_path, is_integral=self.args.SAI, combination_type=self.args.combination_type, epoch=epoch)
        else:
            raise NotImplementedError(f"Submethod {self.args.sub_method} not implemented for path plot")
        self.path_plot_fn = path_plot_fn
    def validation_step(self, batch, batch_idx):
        start_time = time.time()
        NFE = vis_utils.plt_likelihood_dre(self.args, self.model, self.likelihood_fn, 
                                 savefig=self.likelihood_path, epoch=self.current_epoch+1, 
                                 npts=800, device=self.device, steps=self.args.quad_step)
        end_time = time.time()
        val_time = end_time - start_time
        self.path_plot_fn(batch, epoch="final")
        self.log('val_nfe', NFE, prog_bar=True)
        self.log('val_time', val_time, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        epoch = f"solver_{self.args.solver}_steps_{self.args.quad_step}"
        start_time = time.time()
        NFE = vis_utils.plt_likelihood_dre(self.args, self.model, self.likelihood_fn, 
                                 savefig=self.likelihood_path, epoch=epoch, 
                                 npts=800, device=self.device, steps=self.args.quad_step)
        _ = vis_utils.plt_density_ratio(self.args, self.model, self.density_ratio_fn, 
                                 savefig=self.likelihood_path, epoch=epoch, 
                                 npts=800, device=self.device, steps=self.args.quad_step)
        vis_utils.plt_standard_normal_likelihood(self.args, npts=800, savefig=self.likelihood_path, epoch=epoch, device=self.device)
        end_time = time.time()
        test_time = end_time - start_time 
        self.path_plot_fn(batch, epoch="final")
        self.log('test_nfe', NFE, prog_bar=True)
        self.log('test_time', test_time, prog_bar=True)

class DensityEstimationTabularModel(DensityEstimationModel):
    def __init__(self, args, save_path):
        super().__init__(args, save_path)
        self.subtask = "tabular"
        
    def validation_step(self, batch, batch_idx):
        start_time = time.time()
        val_logp, val_nfe = self.likelihood_fn(self.model, batch, joint=self.joint)
        end_time = time.time()
        val_time = end_time - start_time
        
        val_nll = -val_logp.mean()
        self.stage_log("val", stage_nll=val_nll, batch_size=batch.size(0), nfe=val_nfe, time=val_time)
        return {'NLL': val_nll, 'nfe': val_nfe}
    
    def on_validation_epoch_end(self, outputs=None):
        vis_utils.visualize_path_function(self.sde, sample_path=self.sample_path, combination_type=self.args.combination_type, constraint_type=self.args.constraint_type)
    
    def test_step(self, batch, batch_idx):
        start_time = time.time()
        test_logp, test_nfe = self.likelihood_fn(self.model, batch, joint=self.joint, steps=self.args.quad_step)
        end_time = time.time()
        test_time = end_time - start_time
        
        test_nll = -test_logp.mean()  
        self.stage_log("test", stage_nll=test_nll, batch_size=batch.size(0), nfe=test_nfe, time=test_time)
        return {'NLL': test_nll, 'nfe': test_nfe}

class DensityEstimationRealImageModel(DensityEstimationModel):
    def __init__(self, args, save_path):
        super().__init__(args, save_path)
        self.save_hyperparameters()
        self.subtask = args.subtask
        self.noise_level = 0.

        self.data_type = args.data
        self.data_shape = DATA_SHAPES[self.data_type]
        
    def set_basic_functions(self, args):
        self.sde = ImageInterpXt(args=args)
        self.likelihood_fn = self.get_density_fn()
        self.density_ratio_fn = self.get_dr_fn()
        self.train_step_fn = train_step_fns.ScoreMatchingTrainStepFn(args, self.sde)
        self.model = self.build_model()
        self.train_epoch_end_step_fn = lambda : None
        
        self.flow = self.load_pretrained_flow_model()
        self.flow.eval()     
        for param in self.flow.parameters():
            param.requires_grad = False
        
        def flow_fn(batch, train=False): 
            shape = batch.shape
            with torch.no_grad(): 
                x_processed = batch * 255. / 256.   
                x_processed += torch.rand_like(x_processed) / 256.  # Dequantization [0,1]
                x_processed *= 256.  # Scale to [0, 256]
                
                batch, log_det_J = self.flow.transform_to_noise(x_processed, logdet=True, transform=True, train=train)
                batch = batch.reshape(shape)
            return batch, log_det_J
        self.flow_fn = flow_fn
                
    def load_pretrained_flow_model(self,):
        import pickle
        from models.nsf.experiments.images_noise import create_transform
        from models.nsf.nde import distributions, transforms, flows
        
        with open(os.path.join("models/nsf", 'flow_ckpts', 'test_data_stats.p'), 'rb') as fp:
            data_stats = pickle.load(fp)

        device = self.args.device
        train_mean = data_stats['train_mean'].to(device)
        test_mean = data_stats['test_mean'].to(device)
        val_mean = data_stats['val_mean'].to(device)
        train_cov = data_stats['train_cov_cholesky'].to(device)
        val_cov = data_stats['val_cov_cholesky'].to(device)

        # annoying data transforms
        c = 1
        h = w = 28
        spline_params = {
            "apply_unconditional_transform": False,
            "min_bin_height": 0.001,
            "min_bin_width": 0.001,
            "min_derivative": 0.001,
            "num_bins": 128,
            "tail_bound": 3.0
        }
        distribution = distributions.StandardNormal((c * h * w,))

        train_transform, val_transform, transform = create_transform(
            c, h, w, train_mean, val_mean, train_cov, val_cov, levels=2, hidden_channels=64,
            steps_per_level=8, alpha=0.000001, num_bits=8, preprocessing="realnvp_2alpha",
            multi_scale=False, actnorm=True, coupling_layer_type="rational_quadratic_spline",
            spline_params=spline_params, use_resnet=False, num_res_blocks=2,
            resnet_batchnorm=False, dropout_prob=0.0
        )

        # map this "flow" onto the device
        # net = flows.Flow(transform, distribution)
        net = flows.FlowDataTransform(transform, distribution, train_transform, val_transform)

        net = net.to(device)
        # net = torch.nn.DataParallel(net)
        
        return net
        
    def build_model(self):
        hidden_channels = list(map(int, self.args.dims.split("-")))
        # input_shape = (self.args.d,)
        in_channels = self.args.in_channels
        
        if self.decomp:
            return base_models.ImageDecompJointScoreModel(args=self.args,
                                            in_channels=in_channels,
                                            hidden_channels=hidden_channels,
                                            embed_dim=self.args.embed_dim,
                                            nonlinearity=self.args.nonlinearity,
                                            K=self.args.K_decomp,
                                            polynomials=self.args.polynomials,
                                            noise_type=self.args.noise_type,)

        else:
            return base_models.ImageJointScoreModel(args=self.args,
                                        in_channels=in_channels,
                                        hidden_channels=hidden_channels,
                                        embed_dim=self.args.embed_dim,
                                        nonlinearity=self.args.nonlinearity)    
    
    def get_bpd_by_nll(self, nll, x=None, log_det_J=None):
        if x is not None:
            shape = x.shape
            dim = np.prod(shape[1:])
        else:
            dim = np.prod(DATA_SHAPES[self.data_type])

        if log_det_J is not None:
            # NLL(x) = NLL(z) - log|det J|
            nll = nll - log_det_J.mean()
            
        else:
            channels, _, stds = SPECS[self.data_type]
            log_sigma_per_channel = sum(math.log(s) for s in stds)
            pixels_per_channel = dim / channels
            log_det_correction = pixels_per_channel * log_sigma_per_channel
            nll += log_det_correction
        
        return nll, nll / (np.log(2) * dim)
    
    def sampling(self, n_samples, method="HMC", num_steps=100, step_size=0.1, filename="best", **kwargs):
        # 1. Sample from the latent space z ~ p0(z) * r(z)
        with torch.enable_grad():
            z_samples = sampling(self.model, self.sde, self.density_ratio_fn, 
                            n_samples=n_samples, data_shape=self.data_shape, 
                            joint=self.joint, method=method, 
                            num_steps=num_steps, step_size=step_size,
                            device=self.device, normalize_const=self.normalize_const,
                            save_dir=self.sample_path, filename=filename, **kwargs)

        # 2. Inverse Flow: z -> x
        with torch.no_grad():
            # Use .module if using DataParallel, otherwise direct call
            flow_model = self.flow.module if hasattr(self.flow, 'module') else self.flow
            
            # rescale=True ensures output is [-1, 1] 
            samples_x = flow_model.sample(
                z_samples.view(n_samples, -1), 
                context=None, rescale=True, 
                transform=True, train=False     
            )
            
            samples_x = samples_x.view(n_samples, *self.data_shape)

        # 3. Rescale [-1, 1] -> [0, 1] for saving
        samples = (samples_x + 1.) / 2.
        samples = torch.clamp(samples, 0, 1)

        nrow = min(16, math.sqrt(n_samples))
        grid = vutils.make_grid(samples, nrow=nrow, padding=2, normalize=False)
        vutils.save_image(grid, os.path.join(self.sample_path, "samples_best.png"))
        
    def stage_log(self, stage_name, stage_loss=0., stage_nll=0., batch_size=64, nfe=1., bpd=0., time=0.):
        if hasattr(stage_loss, 'item'):
            stage_loss = stage_loss.to(torch.float32)
        stage_nll = stage_nll.to(torch.float32)
        self.log(f"{stage_name}_bpd", bpd, on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        super().stage_log(stage_name=stage_name, stage_loss=stage_loss, stage_nll=stage_nll, batch_size=batch_size, nfe=nfe, time=time)
    
    def training_step(self, batch, batch_idx):
        batch = batch.to(self.device)
        shape = batch.shape
        noise = self.sde.prior_sampling(shape).to(batch)
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
            vis_utils.plot_integral_distributions(self.model, batch, self.sample_path, is_integral=self.args.SAI, combination_type=self.args.combination_type, epoch="final")
        return {'NLL': nll, 'nfe': val_nfe}
     
    def on_validation_epoch_end(self, outputs=None):
        vis_utils.visualize_path_function(self.sde, sample_path=self.sample_path, combination_type=self.args.combination_type, constraint_type=self.args.constraint_type)
        vis_utils.visualize_variance_profiles(self.sde, sample_path=self.sample_path, combination_type=self.args.combination_type)
        # self.sampling(10, method="AIS", num_steps=100, step_size=0.1, filename="best")
        # self.sampling(10, method="HMC", num_steps=500, leapfrog_steps=30, step_size=0.05, filename="best")
        
    def test_step(self, batch, batch_idx):
        batch, log_det_J = self.flow_fn(batch, train=False)
        
        start_time = time.time()
        test_logp, test_nfe = self.likelihood_fn(self.model, batch, joint=False)
        end_time = time.time()
        test_time = end_time - start_time
        test_nll = -test_logp.mean()  
        nll, bpd = self.get_bpd_by_nll(test_nll, batch, log_det_J)
        self.stage_log("test", stage_nll=nll, batch_size=batch.size(0), nfe=test_nfe, bpd=bpd, time=test_time)
        return {'NLL': nll, 'nfe': test_nfe}
    
    def configure_optimizers(self):
        if self.args.sub_method == "kernel":
            self.automatic_optimization = False    # do not need to call optimizer.step() under kernel method
            return []

        if self.args.combination_type == "gpp":
            optimizer = optim.AdamW([
                {'params': self.model.parameters(), 'lr': self.args.lr, 'weight_decay': 1e-5},
                {'params': self.train_step_fn.parameters(), 'lr': self.args.lr, 'weight_decay': 0},
            ])
        else:
            optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=1e-5)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=2000, 
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss", 
            }
        }

# In[] f-divergence estimation
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
        
        # Metrics storage
        self.metric_evaluator = MetricEvaluator()
        self.metrics_db = []
        self.fdiv_db = []
        self.fdiv_std_db = []
        self.fdiv_true_db = []
        self.mae_db = []
        self.nfe_db = []
        self.epoch_db = []
        self.validation_step_outputs = []
        
    def set_prepare_batch(self):
        """Prepare batch data for MI estimation by creating joint and marginal samples."""
        if self.subtask == "mi_realdata":
            if self.data_type == "image":
                def prepare_batch(batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                    x, y, true_mi = batch
                    
                    # Joint samples: concatenate X and Y along channel dimension
                    px = torch.cat([x, y], dim=1)
                    # Marginal samples: concatenate X with shuffled Y
                    batch_size = x.size(0)
                    idx = torch.randperm(batch_size)
                    y_shuffled = y[idx]
                    qx = torch.cat([x, y_shuffled], dim=1)
                    return px, qx, true_mi
            else:  
                # For text data, concatenate along feature dimension
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
                return px, qx, true_mi
        self.prepare_batch = prepare_batch
    
    def set_basic_functions(self, args):
        self.sde = ToyInterpXt(args=args)
        self.density_ratio_fn = self.get_dr_fn()
        self.model = self.build_model()
        self.train_epoch_end_step_fn = lambda : None
        
        if self.args.sub_method == "kernel":
            self.train_step_fn = train_step_fns.KernelDensityRatioTrainStepFn(
                args, sde=self.sde, method=self.subsub_method
            )
            self.train_epoch_end_step_fn = self.model.train_model
            
        elif self.args.sub_method == "neural":
            self.train_step_fn = train_step_fns.NeuralDensityRatioTrainStepFn(
                args, sde=self.sde, method=self.subsub_method
            )
            
        elif self.args.sub_method == "score":
            self.train_step_fn = train_step_fns.ScoreMatchingTrainStepFn(args, self.sde)
            
        else:
            raise ValueError(f"Invalid sub_method {self.args.sub_method}")

    def build_model(self):
        """Constructs the neural network based on decomposition choice."""
        hidden_dims = list(map(int, self.args.dims.split("-")))
        input_shape = (self.d,)
        if self.args.sub_method == "kernel":
            return base_models.KernelDensityRatioModel(
                self.args, sde=self.sde, method=self.subsub_method
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
    
    def get_dr_fn(self):
        return get_density_ratio_fn(self.sde, args=self.args, quad_step=self.args.quad_step)

    def forward(self, t, x):
        return self.model(t, x)
    
    def training_step(self, batch, batch_idx):
        # qx: t=0; px: t=1; density ratio: rx = px/qx
        # px be the shifted, qx be the source
        # px be the joint, qx be the marginal
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
        
        # Calculate metrics
        log_pq, est_nfe = self.density_ratio_fn(self.model, px, joint=self.joint)
        est_fdiv = log_pq.squeeze().mean()
        mae = torch.abs(est_fdiv - true_fdiv).item()
        metrics = self.metric_evaluator.evaluate(true_fdiv, log_pq)
        
        # Log metrics
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
            vis_utils.plot_integral_distributions(self.model, px, self.sample_path, is_integral=self.args.SAI, combination_type=self.args.combination_type, epoch="final")
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

        # Update metrics
        self.metrics_db.append(avg_metrics)
        self.fdiv_db.append(summed_metrics["est_fdiv"])
        self.fdiv_std_db.append(summed_metrics["est_fdiv_std"])
        self.fdiv_true_db.append(true_fdiv)
        self.mae_db.append(summed_metrics["mae"])
        self.nfe_db.append(summed_metrics["est_nfe"])
        self.epoch_db.append(self.current_epoch)
        
        self.validation_step_outputs.clear()
        
        # Save and visualize metrics
        vis_utils.visualize_fdiv(self.args, self.fdiv_db, self.fdiv_true_db, savefig=self.fdiv_path)
        vis_utils.visualize_path_function(self.sde, sample_path=self.sample_path, combination_type=self.args.combination_type, constraint_type=self.args.constraint_type)
        vis_utils.visualize_variance_profiles(self.sde, sample_path=self.sample_path, combination_type=self.args.combination_type)
        with open(os.path.join(self.metrics_dir, 'metrics.p'), 'wb') as f:
            pickle.dump({"true_fdiv": self.fdiv_true_db, "est_fdiv": self.fdiv_db, 
                         "mae": self.mae_db, "nfe": self.nfe_db, 
                         "epoch": self.epoch_db, "metrics": self.metrics_db}, f)
            
    def test_step(self, batch, batch_idx):
        px, _, true_fdiv = self.prepare_batch(batch)
        true_fdiv = true_fdiv.mean().cpu().item() 
        
        # Calculate metrics
        start_time = time.time()
        log_pq, est_nfe = self.density_ratio_fn(self.model, px, joint=self.joint)
        end_time = time.time()
        test_time = end_time - start_time
        
        est_fdiv = log_pq.squeeze().mean()
        mae = torch.abs(est_fdiv - true_fdiv).item()
        metrics = self.metric_evaluator.evaluate(true_fdiv, log_pq)
        
        all_metrics = {
            'true_fdiv': true_fdiv,
            'est_fdiv': est_fdiv,
            'nfe': est_nfe,
            'MAE': mae,
            **metrics
        }
        
        if batch_idx == 0:
            vis_utils.plot_integral_distributions(self.model, px, self.sample_path, is_integral=self.args.SAI, combination_type=self.args.combination_type, epoch="final")
        vis_utils.visualize_variance_profiles(self.sde, sample_path=self.sample_path, combination_type=self.args.combination_type)
        vis_utils.visualize_integrated_variance_profiles(self.sde, sample_path=self.sample_path, combination_type=self.args.combination_type)
        
        for key, value in all_metrics.items():
            if isinstance(value, (int, float)):
                self.log(key, float(value), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            else:
                self.log(key, value, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("test_time", float(test_time), on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True, reduce_fx="sum")

    def configure_optimizers(self):
        if self.args.sub_method == "kernel":
            self.automatic_optimization = False    # do not need to call optimizer.step() under kernel method
            return []
        
        if self.args.combination_type == "gpp":
            optimizer = optim.AdamW([
                {'params': self.model.parameters(), 'lr': self.args.lr, 'weight_decay': 1e-5},
                {'params': self.train_step_fn.parameters(), 'lr': self.args.lr, 'weight_decay': 0},
            ])
        else:
            optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=1e-5)
        return optimizer

class DistributionShiftMeasureModel(DensityRatioEstimationModel):
    def __init__(self, args, save_path, datamodule):
        super().__init__(args, save_path, datamodule)

class MIEstimationModel(DensityRatioEstimationModel):
    def __init__(self, args, save_path, datamodule):
        super().__init__(args, save_path, datamodule)
        self.reload_interval = args.reload_interval
    
    def on_train_epoch_end(self):
        if self.datamodule.__class__.__name__ == "MutualInformationDataModule":
            if self.current_epoch % self.reload_interval == 0:
                self.datamodule.setup()

class MIRealDataModel(DensityRatioEstimationModel):
    """Model for mutual information estimation on real-world datasets."""
    def __init__(self, args, save_path, datamodule):
        super().__init__(args, save_path, datamodule)
        self.save_hyperparameters()
    
    def set_basic_functions(self, args):
        self.sde = ImageInterpXt(args=args)
        self.density_ratio_fn = self.get_dr_fn()
        self.train_step_fn = train_step_fns.ScoreMatchingTrainStepFn(args, self.sde)
        self.data_type = args.data  # 'image', 'text', 'mixture'
        
        # For image data, set appropriate input shape
        if self.data_type == "image":
            self.data_shape = self._get_image_shape(args)
        elif self.data_type == "text":
            self.data_shape = (args.mi_dr,)  # Text embeddings are 1D vectors
        elif self.data_type == "mixture":
            self.data_shape = (args.mi_dr,)   # Mixed data: concatenation of image and text features
        
        if self.data_type in ["image", "mixture"]:
            # Override the model with image-specific architecture if needed
            self.model = self.build_image_model()
        else:
            self.model = self.build_model()
        
    def _get_image_shape(self, args):
        """Calculate the image shape based on representation dimension and channels."""
        img_size = int(np.sqrt(args.mi_dr // args.mi_image_channels))
        mi_image_patches = eval(args.mi_image_patches)
        return (args.mi_image_channels * mi_image_patches[0], img_size, img_size)
    
    def build_image_model(self):
        """Build a model suitable for image data."""
        hidden_channels = list(map(int, self.args.dims.split("-")))
        in_channels = self.data_shape[0] * 2  # Double channels for concatenated X and Y
        
        if self.decomp:
            return base_models.ImageDecompJointScoreModel(
                args=self.args,
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                embed_dim=self.args.embed_dim,
                nonlinearity=self.args.nonlinearity,
                K=self.args.K_decomp,
                polynomials=self.args.polynomials,
                noise_type=self.args.noise_type,
            )
        else:
            return base_models.ImageJointScoreModel(
                args=self.args,
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                embed_dim=self.args.embed_dim,
                nonlinearity=self.args.nonlinearity
            )

class TwoSampleTestModel(DensityRatioEstimationModel):
    def __init__(self, args, save_path, datamodule):
        super().__init__(args, save_path, datamodule)
    
    def calculate_pearson_divergence(self, sample1, sample2):
        n_s = len(sample1)
        n_t = len(sample2)
        r_s = density_ratio_estimator(sample1)
        r_t = density_ratio_estimator(sample2)
        return (0.5 / n_s) * torch.sum(r_s) - (1 / n_t) * torch.sum(r_t) + 0.5
    
    def permutation_test(self, sample1, sample2, K=100):
        original_divergence = self.calculate_pearson_divergence(sample1, sample2)
        combined = torch.cat([sample1, sample2])
        n_s = len(sample1)
        n_t = len(sample2)
        divergences = []
        for _ in range(K):
            permuted = combined[torch.randperm(len(combined))]
            new_sample1 = permuted[:n_s]
            new_sample2 = permuted[n_s:]
            divergence = self.calculate_pearson_divergence(new_sample1, new_sample2)
            divergences.append(divergence)
        divergences = torch.tensor(divergences)
        p_value = torch.sum(divergences >= original_divergence) / K
        return p_value

class ChangePointDetectionModel(DensityRatioEstimationModel):
    def __init__(self, args, save_path, datamodule):
        super().__init__(args, save_path, datamodule)
        
        self.swl = getattr(args, "cpd_swl", 64)
        self.single_test = getattr(args, "single_test", False)
        self.use_simplex = getattr(args, "use_simplex", False)
        self.test_step_outputs = []
        self.validation_step_outputs = []

        # Preprocessing: identity or softmax
        self.preprocess = (lambda x: F.softmax(x, dim=-1)) if self.use_simplex else (lambda x: x)

    def training_step(self, batch, batch_idx):
        px, qx, _ = self.prepare_batch(batch)

        loss_dict = self.train_step_fn(self.model, (qx, px), step=self.global_step)
        loss = loss_dict["loss"]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True, batch_size=px.size(0))
        return loss

    def compute_cpd_statistic(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute change point detection statistic for a full time series.
        
        Args:
            Y: Input time series of shape (T, d).
            
        Returns:
            cp_stat: Change point statistic of shape (T,), where higher values indicate likely change points.
        """
        Y = Y.float()
        T, d = Y.shape
        Y_proc = self.preprocess(Y)  # (T, d)

        # Estimate log r(x) 
        log_ratio, _ = self.density_ratio_fn(self.model, Y_proc, joint=None)
        score = log_ratio  # Use log r(x) as anomaly score

        # Sliding window statistic: |mean(left) - mean(right)|
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

        # Optional: get true change points
        true_cps = None
        if "text" in self.args.data:
            true_cps = [1014, 1608]
        elif "creditcard" in self.args.data:
            true_cps = [1000, 1492]

        vis_utils.visualize_cpd_stat(
            cp_stats=cp_stats,
            window_length=self.swl,
            true_change_points=true_cps,
            save_path=os.path.join(self.metrics_dir, "cpd_stat_summary.pdf"),
            title=f"Change Point Detection: {self.args.data}",
        )

    def validation_step(self, batch, batch_idx):
        Y = batch[0] if isinstance(batch, (list, tuple)) else batch
        if Y.dim() == 3:
            Y = Y.squeeze(0)  # (T, d)

        cp_stat = self.compute_cpd_statistic(Y)
        save_path = os.path.join(self.metrics_dir, f"cpd_stat_val_batch_{batch_idx}.npy")
        np.save(save_path, cp_stat.cpu().numpy())
        self.validation_step_outputs.append({"cp_stat": cp_stat})
    
    def on_validation_epoch_end(self):
        self.on_epoch_end(self.validation_step_outputs)
        self.validation_step_outputs.clear()  # avoid memory leak

    def test_step(self, batch, batch_idx):
        Y = batch[0] if isinstance(batch, (list, tuple)) else batch
        if Y.dim() == 3:
            Y = Y.squeeze(0)  # (T, d)

        cp_stat = self.compute_cpd_statistic(Y)
        save_path = os.path.join(self.metrics_dir, f"cpd_stat_test_batch_{batch_idx}.npy")
        np.save(save_path, cp_stat.cpu().numpy())
        self.test_step_outputs.append({"cp_stat": cp_stat})
    
    def on_test_epoch_end(self):
        self.on_epoch_end(self.test_step_outputs)
        self.test_step_outputs.clear()  # avoid memory leak

# In[]
SUBTASK_MODLE_MODULES = {
    "synthetic2d": DensityEstimation2DSyntheticModel,
    "tabular": DensityEstimationTabularModel,
    "real_image": DensityEstimationRealImageModel,
    "distribution_shift": DistributionShiftMeasureModel,
    "mutual_information": MIEstimationModel,
    "mi_realdata": MIRealDataModel,
    "two_sample_test": TwoSampleTestModel,
    "change_point_detection":ChangePointDetectionModel,
}