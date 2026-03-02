# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import math

from core.utils import divergence_approx, jacobian_frobenius_divergence_approx
from sde import bayesian_path as BayesianPath
from estimators.estimators import secant_to_tangent

class ScoreMatchingTrainStepFn(nn.Module):
    def __init__(self, args, sde):
        super(ScoreMatchingTrainStepFn, self).__init__()
        self.args = args
        self.task = args.task
        self.subtask = args.subtask
        self.device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        
        # base
        self.sde = sde
        self.eps = args.eps
        self.joint = args.joint
        self.condition = args.condition
        self.batch_size = args.batch_size
        self.batch_num = args.batch_num
        self.epochs = args.epochs
        self.total_step = self.epochs * self.batch_num
        self.initial_weight = args.init_weight
        self.final_weight = args.final_weight
        
        self.SAI = args.SAI
        self.gpp = args.gpp
        self.step = 0

        # bridge 
        self.bridge = args.bridge
        self.combination_type = args.combination_type
        
        # decomp
        self.decomp = getattr(args, "decomp", False)
        self.K = args.K_decomp
        self.auxiliary_loss = args.auxiliary_loss
        self.bregman_type = args.bregman_type
        self.clip_tau = torch.tensor(args.clip_tau)
        
        # normalized energy modeling
        self.energy = getattr(args, "energy", False)
        joint = self.joint if not self.energy else True

        self.set_time_sampling_fn(args)
        self.init_loss_and_weight_fns(joint=joint)
        self._init_score_computation()
        self._init_time_gradient_computation()

    def set_time_sampling_fn(self, args):
        self.t_mode = args.t_mode
        if self.t_mode in ["IS", "AIS"]:
            weights_fn=lambda t: self.sde.get_weights_denoise(t)[0]   # weights is inversely proportional to score variance
        else:
            weights_fn = None
        use_interval_annealing=getattr(args, 'IA', False)
        if args.task == "likelihood_estimation":
            annealing_total_steps = min(self.total_step // 2, 50000) 
        else:
            annealing_total_steps = int(self.total_step * 0.5)
            
        update_freq = 1000 if self.gpp else 1000000
        self.t_sampler = BayesianPath.TimeSampler(
                    weights_fn=weights_fn,
                    uniform_eps=self.eps,
                    device=args.device,
                    update_freq=update_freq,
                    grid_size=10000,
                    min_weight=1e-8,
                    t_mode=self.t_mode,
                    aligned_ratio=args.aligned_ratio,
                    use_interval_annealing=use_interval_annealing,
                    annealing_total_steps=annealing_total_steps   
        )
        if self.SAI:
            def time_sampling_fn(batch_size, device="cpu"):
                l, t = self.t_sampler.sample_l_t(batch_size)
                return l.to(device), t.to(device)
        else:
            def time_sampling_fn(batch_size, device="cpu"):
                t = self.t_sampler.sample_t(batch_size).to(device)
                return None, t
        self.time_sampling_fn = time_sampling_fn

    def init_loss_and_weight_fns(self, joint=False):
        if self.condition:
            if self.joint or joint:
                self.loss_fn = self.condition_joint_score_matching
                if self.t_mode == "IS":   # lower variance, easier sampled
                    def weight_fn(time_score_var, data_score_var):
                        return 1., time_score_var / data_score_var     # lambda_t_time = 1. ; lambda_t_data = time_score_var / data_score_var
                else:
                    def weight_fn(time_score_var, data_score_var):
                        return 1. / time_score_var, 1. / data_score_var
                self._weight_fn = weight_fn

            else:
                self.loss_fn = self.condition_time_score_matching
                if self.t_mode == "IS":
                    def weight_fn(time_score_var):
                        return 1.
                else:
                    def weight_fn(time_score_var):
                        return 1. / time_score_var
                self._weight_fn = weight_fn
        else:
            if self.joint or joint:
                self.loss_fn = self.joint_score_matching
            else:
                self.loss_fn = self.time_score_matching
            
            self._weight_fn = self.sde.marginal_var_fn
            self._weight_dt_fn = self.sde.marginal_var_dt_fn
    
    def _init_score_computation(self):
        if self.SAI:
            if self.joint:
                self._compute_scores = self._compute_scores_sai_joint
            else:
                self._compute_scores = self._compute_scores_sai_time
        elif self.energy:
            self._compute_scores = self._compute_scores_energy_joint
        else:
            if self.joint:
                self._compute_scores = self._compute_scores_regular_joint
            else:
                self._compute_scores = self._compute_scores_regular_time
                
    def _init_time_gradient_computation(self):
        if self.decomp:
            self._compute_time_gradient = self._compute_time_gradient_decomp
        else:
            self._compute_time_gradient = self._compute_time_gradient_autograd
            
    # ==================== get scores ====================
    def _compute_scores_sai_joint(self, score_model, x0, x1, t, xt, l):
        """secant alignment identity, joint score"""
        dx_dt = self.sde.marginal_derivative_fn(x0, x1, t, *self.sde.get_coefficients_and_derivatives(t)[:4])
        score_x, score_t = secant_to_tangent(score_model, xt, dx_dt, l, t, joint=True)
        return score_x, score_t
    
    def _compute_scores_sai_time(self, score_model, x0, x1, t, xt, l):
        """Secant alignment identity, time score"""
        dx_dt = self.sde.marginal_derivative_fn(x0, x1, t, *self.sde.get_coefficients_and_derivatives(t)[:4])
        score_t = secant_to_tangent(score_model, xt, dx_dt, l, t, joint=False)
        return score_t
    
    def _compute_scores_regular_joint(self, score_model, x0, x1, t, xt, l):
        return score_model(t, xt)
    
    def _compute_scores_regular_time(self, score_model, x0, x1, t, xt, l):
        return score_model(t, xt).squeeze()
    
    def _compute_scores_energy_joint(self, score_model, x0, x1, t, xt, l):
        xt = xt.requires_grad_(True)
        t = t.requires_grad_(True)
        
        energy = score_model(t, xt)  # [batch, 1]
        v_output = torch.ones_like(energy)  # [batch, 1]

        grads_t, grads_x = torch.autograd.grad(
            outputs=energy,
            inputs=(t, xt),
            grad_outputs=v_output,
            create_graph=True,
            retain_graph=True
        )
        return -grads_x, -grads_t
    
    # ==================== time score gradient w.r.t. t ====================
    def _compute_time_gradient_decomp(self, score_model, score_t, t, xt, fk):
        return score_model.time_gradient(t, xt, f_k=fk)
    
    def _compute_time_gradient_autograd(self, score_model, score_t, t, xt, fk):
        grad_outputs = torch.ones_like(score_t) / score_t.size(0)
        return autograd.grad(score_t, t, grad_outputs=grad_outputs, create_graph=True)[0]
    
    def adaptive_loss_weighting(self, loss_list, temperature=5.0, eps=1e-8):
        """
        This method automatically assigns higher weights to losses with smaller magnitudes,
        effectively balancing different objectives without manually tuning loss coefficients.

        Args:
            loss_list (List[Tensor]): List of per-batch loss tensors, each with shape [B, ...].
            temperature (float): Softmax temperature. Higher values produce more uniform weights,
                                lower values make weighting more "winner-takes-all".
            eps (float): Small constant for numerical stability to avoid division by zero.
        """
        mean_losses = [l.squeeze().mean() for l in loss_list]

        detached_means = torch.stack([ml.detach() for ml in mean_losses], dim=0)
        weights = 1.0 / (detached_means + eps)
        weights = F.softmax(weights / temperature, dim=0)

        total_loss = torch.sum(weights * torch.stack(mean_losses, dim=0))
        return total_loss
    
    def get_current_weight(self, step, total_epochs=None, initial_weight=None, final_weight=None, strategy="cosine"):
        if total_epochs is None:
            total_epochs = self.epochs
        if initial_weight is None:
            initial_weight = self.initial_weight
        if final_weight is None:
            final_weight = self.final_weight
        epoch = step // self.batch_num

        if strategy == "linear":
            """Linear adjustment strategy. Adjust the power to accelerate the linear decreasing speed. Here, the power is set to 2, which can be adjusted as needed."""
            power = 1
            current_weight = initial_weight - (initial_weight - final_weight) * (torch.tensor(epoch / total_epochs).pow(power))
        elif strategy == "exponential":
            """Exponential adjustment strategy. Dynamically adjust the decay rate according to the number of iterations. The coefficient 0.001 here can be adjusted according to the actual situation."""
            decay_rate = (epoch / total_epochs)  # 0.001 * 
            current_weight = initial_weight * torch.exp(-torch.tensor(decay_rate))
            current_weight = max(current_weight, final_weight).item()
        elif strategy == "step":
            """Step adjustment strategy. Reasonably set the step interval and the weight value decreased each time according to the number of iterations."""
            step_size = 1000  # int(math.sqrt(total_epochs))
            step_weight = (initial_weight - final_weight) / (total_epochs // step_size)
            current_weight = initial_weight - (epoch // step_size) * step_weight
            current_weight = max(current_weight, final_weight)
        elif strategy == "cosine":
            """Cosine annealing adjustment strategy. Adjust the period according to the number of iterations. The coefficient 2 here can be adjusted according to the actual situation."""
            T = total_epochs 
            current_weight = final_weight + 0.5 * (initial_weight - final_weight) * (1 + torch.cos(torch.tensor(math.pi * 0.5 * epoch / T))).item()
        else:
            raise ValueError("The weight adjustment strategy is not supported. Please choose 'linear', 'exponential','step' or 'cosine' strategy.")

        return current_weight
    
    def _compute_coefficients_and_sample(self, x0, x1, t, with_grad=False):
        """evaluate coefficients and sample xt"""
        alpha_t, beta_t, d_alpha_dt, d_beta_dt = self.sde.get_coefficients_and_derivatives(t)
        xt = self.sde.marginal_sample(x0, x1, t, alpha_t=alpha_t, beta_t=beta_t)
        
        if with_grad:
            xt = xt.requires_grad_(True)
            t = t.requires_grad_(True)
            
        return alpha_t, beta_t, d_alpha_dt, d_beta_dt, xt, t
    
    def _precompute_norms(self, x0, x1):
        # return x0_norm_sq, x1_norm_sq, x0x1_norm_sq
        return self.sde.inner_product(x0, x0), self.sde.inner_product(x1, x1), self.sde.inner_product(x0, x1)
        
    def _apply_kmm_regularization(self, loss, score_t, x0, x1, time_score_var, batch_size):
        # used in KMM param...
        t = torch.rand(batch_size, 1).to(x0) * (1 - 2 * self.eps) + self.eps
        # score_reg = score_t ** 2
        score_reg = 0.
        variance = self.sde.get_score_var(t, x0, x1)[0]
        elbo = self.sde.gaussian_process.elbo(x0, x1, t, variance=variance, score_reg=score_reg)
        
        weight = 1. / time_score_var.mean()
        return self.adaptive_loss_weighting([loss * weight, -elbo * weight])
    
    def time_score_matching(self, score_model, x0, x1, t0, t1, t, l=None, decomp=False, batch_size=None, dim=1):    
        alpha_t, beta_t, d_alpha_dt, d_beta_dt, xt, t = self._compute_coefficients_and_sample(x0, x1, t, with_grad=True)
        with torch.no_grad():
            lambda_t = self._weight_fn(t, alpha_t=alpha_t, beta_t=beta_t)
            lambda_dt = self._weight_dt_fn(t, alpha_t=alpha_t, beta_t=beta_t, d_alpha_dt=d_alpha_dt, d_beta_dt=d_beta_dt)
            lambda_t0 = self._weight_fn(t0)
            lambda_t1 = self._weight_fn(t1)
        
        score_t = self._compute_scores(score_model, x0, x1, t, xt, l)
        fk_xt = score_model.get_fk_reg() if hasattr(score_model, 'get_fk_reg') else None
        assert score_t.ndim <= 2, "score_t should be at most 2D"
        score_t = score_t[(...,) + (None,) * (x0.ndim - score_t.ndim)]
        
        xt_score_dt = self._compute_time_gradient(score_model, score_t, t, xt, fk_xt)
        
        ####################
        # Boundary condition
        ####################
        boundary_0 = (score_model(t0, x0)).squeeze() * lambda_t0
        fk_x0 = score_model.get_fk_reg() if hasattr(score_model, 'get_fk_reg') else None

        boundary_1 = (score_model(t1, x1)).squeeze() * lambda_t1
        fk_x1 = score_model.get_fk_reg() if hasattr(score_model, 'get_fk_reg') else None
            
        boundary_loss_terms = {"boundary_0": boundary_0, "boundary_1": boundary_1, 
                               "fk_x0": fk_x0, "fk_x1": fk_x1}
        
        ####################
        # Score-matching loss
        ####################
        term1 = xt_score_dt * lambda_t
        term2 = score_t * lambda_dt
        term3 = 0.5 * score_t ** 2 * lambda_t

        loss = boundary_0 - boundary_1 + term1 + term2 + term3
        return {"loss": loss, "score_t": score_t, "xt_score_dt": xt_score_dt, 
                "term3": term1, "term4": term2, "term5": term3,
                "boundary_loss_terms": boundary_loss_terms, "fk_xt": fk_xt,}
        
    def joint_score_matching(self, score_model, x0, x1, t0, t1, t, l=None, decomp=False, batch_size=None, dim=1):
        alpha_t, beta_t, d_alpha_dt, d_beta_dt, xt, t = self._compute_coefficients_and_sample(x0, x1, t, with_grad=True)
        
        with torch.no_grad():
            # lambda_t = t * (1 - t)
            # lambda_dt = 1 - 2 * t
            # lambda_t0 = t0 * (1 - t0)
            # lambda_t1 = t1 * (1 - t1)
            lambda_t = self._weight_fn(t, alpha_t=alpha_t, beta_t=beta_t)
            lambda_dt = self._weight_dt_fn(t, alpha_t=alpha_t, beta_t=beta_t, d_alpha_dt=d_alpha_dt, d_beta_dt=d_beta_dt)
            lambda_t0 = self._weight_fn(t0)
            lambda_t1 = self._weight_fn(t1)
            lambda_t_time = lambda_t_data = lambda_t
        
        score_x, score_t= self._compute_scores(score_model, x0, x1, t, xt, l)
        fk_xt = score_model.get_fk_reg() if hasattr(score_model, 'get_fk_reg') else None 
        assert score_t.ndim <= 2, "score_t should be at most 2D"
        score_t = score_t[(...,) + (None,) * (x0.ndim - score_t.ndim)]
        
        xt_score_dt = self._compute_time_gradient(score_model, score_t, t, xt, fk_xt)

        ####################
        # Boundary condition
        ####################
        boundary_0 = (score_model(t0, x0)[-1]) * lambda_t0
        fk_x0 = score_model.get_fk_reg() if hasattr(score_model, 'get_fk_reg') else None
            
        boundary_1 = (score_model(t1, x1)[-1]) * lambda_t1
        fk_x1 = score_model.get_fk_reg() if hasattr(score_model, 'get_fk_reg') else None
        
        boundary_loss_terms = {"boundary_0": boundary_0, "boundary_1": boundary_1, 
                               "fk_x0": fk_x0, "fk_x1": fk_x1}
        
        ####################
        # Score-matching loss
        ####################
        term3 = xt_score_dt * lambda_t_time
        term4 = score_t * lambda_dt
        time_loss = boundary_0 - boundary_1 + term3 + term4
        
        # data score-matching
        grad1 = torch.cat([score_x.view((batch_size,-1)), score_t.view((batch_size,-1))], dim=-1)
        ssm_term1 = 0.5 * (torch.sum(grad1 * grad1, dim=-1, keepdim=True))
        ssm_term2 = divergence_approx(score_x, xt)   
        ssm_loss = (ssm_term1 + ssm_term2) * lambda_t_data
        
        loss = ssm_loss + time_loss
        return {"loss": loss, "ssm_loss": ssm_loss, "ssm_term1": ssm_term1, "ssm_term2": ssm_term2,
                "time_loss": time_loss, "term3": term3, "term4": term4,
                "boundary_loss_terms": boundary_loss_terms, "fk_xt": fk_xt,}
    
    def condition_time_score_matching(self, score_model, x0, x1, t0, t1, t, l=None, decomp=False, batch_size=None, dim=1):
        x0_norm_sq, x1_norm_sq, x0x1_norm_sq = self._precompute_norms(x0, x1)
        
        with torch.no_grad():
            alpha_t, beta_t, d_alpha_dt, d_beta_dt, xt, t = self._compute_coefficients_and_sample(x0, x1, t)
        
            marginal_var = self.sde.marginal_var_fn(t, alpha_t, beta_t)   # used for stable training
            marginal_std = marginal_var.sqrt()
            
            time_score_var = self.sde.get_score_var(t, x0, x1, x0_norm_sq, x1_norm_sq, x0x1_norm_sq, alpha_t, beta_t, d_alpha_dt, d_beta_dt)[0]
            target_cond_time_score = self.sde.get_score_target(t, x0, x1, x0_norm_sq, x1_norm_sq, x0x1_norm_sq, alpha_t, beta_t, d_alpha_dt, d_beta_dt, dim)

            lambda_t_time = self._weight_fn(time_score_var)
            self.sde.set_z()

        score_t = self._compute_scores(score_model, x0, x1, t, xt, l)
        fk_xt = score_model.get_fk_reg() if hasattr(score_model, 'get_fk_reg') else None 
        assert score_t.ndim <= 2, "score_t should be at most 2D"
        score_t = score_t[(...,) + (None,) * (x0.ndim - score_t.ndim)]
        
        ####################
        # Time score-matching loss
        ####################
        time_diff = marginal_std * (score_t - target_cond_time_score)
        time_loss = lambda_t_time * time_diff**2 / (marginal_var + 1e-5)
        loss = time_loss
        
        ####################
        # KMM (gpp, Gaussian process prior)
        ####################
        if self.gpp:
            loss = self._apply_kmm_regularization(loss, score_t, x0, x1, time_score_var, batch_size)

        return {"loss": loss, "target_cond_time_score": target_cond_time_score, "fk_xt": fk_xt, 
                "lambda_t_time": lambda_t_time, "variance": time_score_var}
    
    def condition_joint_score_matching(self, score_model, x0, x1, t0, t1, t, l=None, decomp=False, batch_size=None, dim=1):
        x0_norm_sq, x1_norm_sq, x0x1_norm_sq = self._precompute_norms(x0, x1)
         
        with torch.no_grad():
            alpha_t, beta_t, d_alpha_dt, d_beta_dt, xt, t = self._compute_coefficients_and_sample(x0, x1, t)

            marginal_var = self.sde.marginal_var_fn(t, alpha_t, beta_t)   # used for stable training
            marginal_std = marginal_var.sqrt()
            
            time_score_var, data_score_var = self.sde.get_score_var(t, x0, x1, x0_norm_sq, x1_norm_sq, x0x1_norm_sq, alpha_t, beta_t, d_alpha_dt, d_beta_dt)
            target_cond_time_score, target_cond_data_score = self.sde.get_score_target(t, x0, x1, x0_norm_sq, x1_norm_sq, x0x1_norm_sq, alpha_t, beta_t, d_alpha_dt, d_beta_dt, dim)

            lambda_t_time, lambda_t_data = self._weight_fn(time_score_var, data_score_var)
            self.sde.set_z()
        
        score_x, score_t= self._compute_scores(score_model, x0, x1, t, xt, l)
        fk_xt = score_model.get_fk_reg() if hasattr(score_model, 'get_fk_reg') else None 
        assert score_t.ndim <= 2, "score_t should be at most 2D"
        score_t = score_t[(...,) + (None,) * (x0.ndim - score_t.ndim)]
        
        ####################
        # Score-matching loss
        ####################
        time_diff = marginal_std * (score_t - target_cond_time_score)
        time_loss = lambda_t_time * time_diff**2 / (marginal_var + 1e-5)
        
        data_diff = marginal_std * (score_x - target_cond_data_score)
        ssm_loss_unreduced =  data_diff**2 / (marginal_var + 1e-5)
        ssm_loss = lambda_t_data * torch.mean(ssm_loss_unreduced, dim=tuple(range(1, data_diff.ndim)), keepdim=True)
        
        loss = time_loss + ssm_loss
        
        ####################
        # KMM (gpp, Gaussian process prior)
        ####################
        if self.gpp:
            loss = self._apply_kmm_regularization(loss, score_t, x0, x1, time_score_var, batch_size)
            
        return {"loss": loss, "ssm_loss": ssm_loss, "time_loss": time_loss, "fk_xt": fk_xt, 
                "target_cond_time_score": target_cond_time_score, "target_cond_data_score": target_cond_data_score,
                "lambda_t_time": lambda_t_time, "variance": time_score_var, "score_t": score_t}
    
    def forward(self, score_model, samples, step=0, log_det_J=0.):
        # How to set x0 and x1? x0: px, noise; x1: qx, data
        # our target is to estimate log r(x) = log qx/px = int_{0}^{1} time_score dt
        # so Bregman divergence maximize log r(x1) and minimize log r(x0)
        eps = self.eps
        decomp = self.decomp

        x0, x1 = samples 
        device = x0.device
        batch_size = x0.size(0)
        dim = x0.numel() // batch_size
        
        l, t = self.time_sampling_fn(batch_size=batch_size, device=device)
        
        x0 = x0.to(device)
        x1 = x1.to(device)
        t0 = torch.zeros_like(t) + eps
        t1 = torch.ones_like(t) - eps
        
        all_loss_terms = self.loss_fn(score_model, x0, x1, t0, t1, t, l=l, decomp=decomp, batch_size=batch_size, dim=dim)
        loss = all_loss_terms["loss"].view((batch_size, 1))
        self.step += 1
        
        nll = torch.zeros_like(t)
        if decomp:
            reg_loss = 0.
            # base_weight = self.get_current_weight(step)
            base_weight = 1. / self.K 
 
            # fk_x0 = all_loss_terms["boundary_loss_terms"]["fk_x0"]
            # fk_x1 = all_loss_terms["boundary_loss_terms"]["fk_x1"]
            fk_xt = all_loss_terms["fk_xt"]

            if self.auxiliary_loss:
                # bregman divergence
                # logr0 = score_model.compute_weighted_sum(x0, f_k=fk_x0)
                # logr1 = score_model.compute_weighted_sum(x1, f_k=fk_x1)
                logr0 = score_model.compute_weighted_sum(x0)
                logr1 = score_model.compute_weighted_sum(x1)
                bregman = self.bregman_div_loss(logr0, logr1, bregman_type=self.bregman_type)
                reg_loss += bregman
                
                # if self.task == "likelihood_estimation":
                #     nll = - (logr1.view((batch_size, 1)) + self.sde.prior_logp(x1).view((batch_size, 1))).mean()
                #     # nll = - score_model.likelihood_estimation(x1, f_k=fk_x1).view((batch_size, 1))
                    
                #     # bpd = nll / (np.log(2) * np.prod(x1.shape[1:]))  # bpd
                #     reg_loss += (nll - log_det_J.mean()) / torch.tensor((np.log(2) * np.prod(x1.shape[1:]))).to(x1)

            #     elif self.task == "fdiv_estimation":
            #         pass     
            #         # MINE loss
            #         # joint_expectation = logr1.mean(dim=0)   
            #         # marginal_expectation = torch.log(torch.mean(torch.exp(logr0), dim=0))         
            #         # mine_loss = - (joint_expectation - marginal_expectation)
            #         # loss += weight * mine_loss 
            
            # # loss = loss + reg_loss #/ self.K 
            
            loss = loss + reg_loss
            

            # if step % 1 == 0:
            #     def get_avg_and_format(loss):
            #         avg = loss.mean().detach().item()
            #         return "{:.4f}".format(avg)
            #     print("======================================")
            #     loss_avg = get_avg_and_format(loss)
            #     ssm_term1_avg = get_avg_and_format(ssm_term1)
            #     ssm_term2_avg = "{:.4f}".format(ssm_term2.mean().detach().item())
            #     term1_avg = "{:.4f}".format(term1.mean().detach().item())
            #     term2_avg = "{:.4f}".format(term2.mean().detach().item())
            #     term3_avg = "{:.4f}".format(term3.mean().detach().item())
            #     term4_avg = "{:.4f}".format(term4.mean().detach().item())
            #     l12_loss_avg = "{:.4f}".format(l12_loss.mean().detach().item())
            #     # bregman_avg = "{:.4f}".format(bregman.mean().detach().item())
            #     # # mine_loss_avg = "{:.4f}".format(mine_loss.mean().detach().item())
                
            #     score_x_norm = 0.5 * (torch.sum(score_x.view((batch_size,-1)) * score_x.view((batch_size,-1)), dim=-1)).view((batch_size,-1))
            #     score_t_norm = 0.5 * (torch.sum(score_t * score_t, dim=-1)).view((batch_size,-1))
            #     score_x_norm_avg = get_avg_and_format(score_x_norm)
            #     score_t_norm_avg = get_avg_and_format(score_t_norm)
            #     clip_loss_avg = "{:.4f}".format(clip_loss.mean().detach().item())
            #     logr0 = get_avg_and_format(logr0)
            #     logr1 = get_avg_and_format(logr1)
            #     # print(f"score_x: {score_x_norm_avg}, score_t: {score_t_norm_avg}, breg: {bregman_avg}, clip: {clip_loss_avg}, l12_loss={l12_loss_avg}, logr0: {logr0}, logr1: {logr1}")

            #     elbo_avg = get_avg_and_format(elbo)
            #     print(f"score_x: {score_x_norm_avg}, score_t: {score_t_norm_avg}, logr0: {logr0}, logr1: {logr1}, elbo: {elbo_avg}")
                
            #     # print("MINE loss: ", mine_loss_avg, joint_expectation, marginal_expectation)
            #     # print(bregman.detach().mean().cpu().item())
            #     # Breg={bregman.detach().mean().cpu().item()}, 
            #     print(f"Step {step}: loss={loss_avg}, ssm_term1={ssm_term1_avg}, ssm_term2={ssm_term2_avg}, term1={term1_avg}, term2={term2_avg}, term3={term3_avg}, term4={term4_avg}")
                
            score_model.set_fk()
            score_model.Pk.set_cached_sigma()
            return {'loss': loss.mean(), 'nll': nll.mean()}
        
        else:
            # if step % 1 == 0:
            #     def get_avg_and_format(loss):
            #         avg = loss.mean().detach().item()
            #         return "{:.4f}".format(avg)
            #     print("======================================")
            #     # loss_avg = get_avg_and_format(loss)
            #     # ssm_term1_avg = get_avg_and_format(ssm_term1)
            #     # ssm_term2_avg = "{:.4f}".format(ssm_term2.mean().detach().item())
            #     # term1_avg = "{:.4f}".format(term1.mean().detach().item())
            #     # term2_avg = "{:.4f}".format(term2.mean().detach().item())
            #     # term3_avg = "{:.4f}".format(term3.mean().detach().item())
            #     # term4_avg = "{:.4f}".format(term4.mean().detach().item())
            #     # l12_loss_avg = "{:.4f}".format(l12_loss.mean().detach().item())
            #     # bregman_avg = "{:.4f}".format(bregman.mean().detach().item())
            #     # # mine_loss_avg = "{:.4f}".format(mine_loss.mean().detach().item())
                
            #     # score_x_norm = 0.5 * (torch.sum(score_x.view((batch_size,-1)) * score_x.view((batch_size,-1)), dim=-1)).view((batch_size,-1))
            #     # score_t_norm = 0.5 * (torch.sum(score_t * score_t, dim=-1)).view((batch_size,-1))
            #     # score_x_norm_avg = get_avg_and_format(score_x_norm)
            #     # score_t_norm_avg = get_avg_and_format(score_t_norm)
            #     # clip_loss_avg = "{:.4f}".format(clip_loss.mean().detach().item())
            #     # logr0 = get_avg_and_format(logr0)
            #     # logr1 = get_avg_and_format(logr1)
            #     # print(f"score_x: {score_x_norm_avg}, score_t: {score_t_norm_avg}, breg: {bregman_avg}, clip: {clip_loss_avg}, l12_loss={l12_loss_avg}, logr0: {logr0}, logr1: {logr1}")

            #     elbo_avg = get_avg_and_format(-elbo)
            #     print(f"elbo: {elbo_avg}")
            
            # likelihood_fn = self.get_likelihood_fn(self.sde, rtol=self.args.rtol, atol=self.args.atol, solver=self.args.solver, eps=self.args.eps)
            # logp, nfe = likelihood_fn(score_model, x0, self.joint)
            # nll = -logp
            return {'loss': loss.mean(), 'nll': nll.mean()}
    
    def sparsity_condition(self, sigma):
        # l1 and l2, elastic regularization
        width_reg_l1 = sigma.sum().squeeze()
        # width_reg_l2 = sigma.pow(2).mean().squeeze()  
        return width_reg_l1 #+ width_reg_l2 
    
    def inverse_sparsity_condition(self, sigma):
        return self.sparsity_condition(1./sigma)
    
    def soft_clip_loss(self, x, tau, max_clip=5.0):
        diff = torch.relu(torch.abs(x) - tau)#.clamp(max=max_clip)
        # diff = max_clip * torch.tanh(diff / max_clip)
        return torch.sum(diff.pow(2) + diff.abs(), dim=1)
    
    def bregman_div_loss(self, logr0, logr1, bregman_type="logistic", eps=1e-8, bound=5., gamma=0.1):
        """Compute BD loss, we need: smaller logr0 and larger logr1"""
        logr0 = torch.clamp(logr0, min=-20, max=20)
        logr1 = torch.clamp(logr1, min=-20, max=20)
        if bregman_type == "logistic":
            # Logistic sigmoid: φ(z) = log(1 + exp(-z))
            # BD = -E_p0[-log{1+r0}] - E_p1[log\frac{r1}{1+r1}]
            # term1 = - (- torch.log1p(r0))
            # term2 = - (logr1 - torch.log1p(r1))
            
            term1 = - F.logsigmoid(-logr0)  # .clamp(max=1.)
            term2 = - F.logsigmoid(logr1)
            
            # term1 = - F.logsigmoid(-logr0.clamp(max=0.))  # .clamp(max=1.)
            # term2 = - F.logsigmoid(logr1.clamp(max=bound))

            # # gamma=0.1
            # denom_p0 = 1 + gamma * F.relu(logr0)  # 仅对 r0 > 1 增强
            # denom_p1 = 1 + gamma * F.relu(-logr1)  # 仅对 r1 < 1 增强
            # w0 = 1 / denom_p0
            # w1 = 1 / denom_p1
            # term1 = - F.logsigmoid(-logr0) * w0
            # term2 = - F.logsigmoid(logr1) * w1
            
        else:
            # Compute r with numerical stability
            r0 = torch.exp(logr0.clamp(max=1.))  # shape: [batch_size, 1]
            r1 = torch.exp(logr1.clamp(max=bound))  # shape: [batch_size, 1]
            
            if bregman_type == "kl":
                # KL divergence: φ(z) = z log z - z
                # BD = E_p0[r log r - r] - E_p1[log r] + C
                term1 = r0 * logr0 - r0  # [batch_size, 1]
                term2 = -logr1                # [batch_size, 1]
                
            elif bregman_type == "pearson":
                # Pearson χ² divergence: φ(z) = 0.5(z-1)^2
                # BD = E_p0[0.5 r^2 - r] - E_p1[r - 1] + C
                term1 = 0.5 * r0**2 - r0          # [batch_size, 1]
                term2 = -(r1 - 1)                   # [batch_size, 1]
                
            elif bregman_type == "itakura_saito":
                # Itakura-Saito divergence: φ(z) = -log z
                # BD = E_p0[-1 + log r] - E_p1[-1/r] + C
                term1 = -1 + logr0           # [batch_size, 1]
                term2 = 1 / (r1 + eps)              # [batch_size, 1]
            
            else:
                raise ValueError(f"Unknown Bregman type: {bregman_type}")
        loss = term1 + term2
        return loss.view(-1,1)  # shape: [batch_size, 1]

