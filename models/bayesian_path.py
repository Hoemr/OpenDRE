import os
import copy
import math
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.func import vmap, jacrev
from torch.distributions import Beta

def stopgrad(x):
    return x.detach()

class BayesianTrajectory(nn.Module):
    """Base class for Bayesian trajectory optimization"""
    
    def __init__(self, n_components=3, constraint_type="spherical", eps=1e-5, grid_size=200):
        super().__init__()
        self.n_components = n_components
        self.constraint_type = constraint_type
        self.eps = eps
        self.register_buffer("grid", torch.linspace(eps, 1-eps, grid_size).view(-1, 1))
        
        # Common parameters
        self.logits = nn.Parameter(torch.zeros(n_components))
        self.register_buffer("prior_logits", torch.zeros(n_components))
        
        # Common buffers
        self.register_buffer("sqrt_2pi", torch.tensor(np.sqrt(2 * np.pi)))
        self.register_buffer("sqrt_2", torch.tensor(np.sqrt(2)))
        
        self.set_beta_and_dot_beta_fn()
        
    def set_beta_and_dot_beta_fn(self):
        if self.constraint_type == "affine":
            def get_beta_and_dot_beta(alpha, dot_alpha):
                beta = 1 - alpha
                dot_beta = -dot_alpha
                return beta, dot_beta
        elif self.constraint_type == "spherical":
            def get_beta_and_dot_beta(alpha, dot_alpha):
                beta = torch.sqrt(1 - alpha**2)
                dot_beta = -alpha * dot_alpha / (beta + self.eps)
                return beta, dot_beta
        else:
            raise NotImplementedError("constraint_type must be affine or spherical")
        self.get_beta_and_dot_beta = get_beta_and_dot_beta
    
    def get_params(self, mode="posterior"):
        """Abstract method to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement get_params")
    
    def distribution_cdf(self, t, weights, *params):
        """Abstract method for CDF computation"""
        raise NotImplementedError("Subclasses must implement distribution_cdf")
    
    def distribution_pdf(self, t, weights, *params):
        """Abstract method for PDF computation"""
        raise NotImplementedError("Subclasses must implement distribution_pdf")
    
    def forward(self, t, mode="posterior", return_derivs=False):
        """Compute alpha, beta and their derivatives at time t"""
        params = self.get_params(mode=mode)
        weights = params[0]
        distribution_params = params[1:]
        
        Ft = self.distribution_cdf(t, weights, *distribution_params)
        
        # Compute alpha, beta and their derivative
        alpha = 1 - Ft
        pdf_t = self.distribution_pdf(t, weights, *distribution_params)
        dot_alpha = -pdf_t
        alpha = torch.clamp(alpha, self.eps, 1 - self.eps)
        beta, dot_beta = self.get_beta_and_dot_beta(alpha, dot_alpha)

        output = {
            "alpha": alpha,
            "beta": beta,
        }
        
        if return_derivs:
            output.update({
                "dot_alpha": dot_alpha,
                "dot_beta": dot_beta,
            })
            
        return output
    
    def sample_at_time(self, t, mode="posterior", return_derivs=False, mean=False):
        """Sample trajectory values at specific time points"""
        return self(t, mode=mode, return_derivs=return_derivs)
    
    def sample_trajectories(self, mode="posterior", return_derivs=False):
        """Sample full trajectories on a grid"""
        return self(self.grid, mode=mode, return_derivs=return_derivs)
    
    def compute_variance(self, t, x0, x1):
        """Compute variance of time score at time t"""
        samples = self(t, return_derivs=True)
        
        alpha = samples["alpha"]
        dot_alpha = samples["dot_alpha"]
        dot_beta = samples["dot_beta"]
        
        # Compute squared norm of x1
        x1_norm_sq = x1.flatten(1).pow(2).sum(dim=1, keepdim=True)
        
        # Compute variance components
        dim = x1.numel() / x1.size(0)
        term1 = 2 * dim * dot_alpha**2
        term2 = x1_norm_sq * dot_beta**2
        
        # Compute variance with protection
        variance = (term1 + term2) / (alpha**2 + self.eps)
        return torch.clamp(variance, min=1e-5, max=1e6)
    
    def elbo(self, x0, x1, t, variance=None, score_reg=0.):
        """Evidence Lower Bound with variance regularization"""
        if variance is None:
            variance = self.compute_variance(t, x0, x1)
        return - (variance + score_reg)
    
    def stopgrad(self, x):
        """Utility function to stop gradients"""
        return x.detach()
    
class GMMBayesianTrajectory(BayesianTrajectory):
    """Bayesian trajectory optimization with GMM-CDF parameterization"""
    
    def __init__(self, n_components=3, constraint_type="spherical", eps=1e-5, grid_size=200, mean_range=(0.0, 1.0)):
        super().__init__(n_components, constraint_type, eps, grid_size)
        
        # GMM specific parameters
        self.means = nn.Parameter(torch.linspace(mean_range[0] + 0.1, mean_range[1] - 0.1, n_components))
        self.log_stds = nn.Parameter(torch.ones(n_components) * np.log(0.3))
        
        # GMM prior parameters
        self.register_buffer("prior_means", torch.linspace(mean_range[0] + 0.1, mean_range[1] - 0.1, n_components))
        self.register_buffer("prior_log_stds", torch.ones(n_components) * np.log(0.3))
        
        # Parameter constraints
        self.mean_min = mean_range[0] - 0.2
        self.mean_max = mean_range[1] + 0.2
        self.std_min = 0.05
        self.std_max = 2.0
    
    def get_params(self, mode="posterior"):
        """Convert unconstrained parameters to constrained ones"""
        if mode == "posterior":
            weights = F.softmax(self.logits, dim=-1)
            means = self.means
            stds = F.softplus(self.log_stds) + self.eps
            return weights, means, stds
        else:
            weights = F.softmax(self.prior_logits, dim=-1)
            means = self.prior_means
            stds = F.softplus(self.prior_log_stds) + self.eps
            return weights, means, stds
    
    def distribution_cdf(self, t, weights, means, stds):
        """Compute the CDF of the GMM at time t"""
        z = (t - means) / (stds + self.eps)
        z = z.clamp(-10, 10)
        component_cdfs = 0.5 * (1 + torch.erf(z / self.sqrt_2))
        cdf_vals = (weights * component_cdfs).sum(dim=1, keepdim=True).clamp(self.eps, 1.0 - self.eps)
        return cdf_vals
    
    def distribution_pdf(self, t, weights, means, stds):
        """Compute the PDF of the GMM at time t"""
        exponent = -0.5 * ((t - means) / (stds + self.eps))**2
        exponent = torch.clamp(exponent, -50.0, 50.0)
        pdf_vals = weights * torch.exp(exponent) / (stds * self.sqrt_2pi + self.eps)
        pdf_vals = pdf_vals.sum(dim=1, keepdim=True)
        return pdf_vals
    
    def forward(self, t, mode="posterior", return_derivs=False):
        """Compute alpha, beta and their derivatives at time t"""
        params = self.get_params(mode=mode)
        weights = params[0]
        distribution_params = params[1:]
        
        Ft = self.distribution_cdf(t, weights, *distribution_params)
        
        t_zeros = torch.zeros_like(t).to(t)
        t_ones = torch.ones_like(t).to(t)
        F0 = self.distribution_cdf(t_zeros, weights, *distribution_params)
        F1 = self.distribution_cdf(t_ones, weights, *distribution_params)
        
        # Compute alpha and its derivative
        alpha = 1 - (Ft - F0) / (F1 - F0 + self.eps)
        pdf_t = self.distribution_pdf(t, weights, *distribution_params)
        dot_alpha = -pdf_t / (F1 - F0 + self.eps)
        alpha = torch.clamp(alpha, self.eps, 1 - self.eps)
        
        beta, dot_beta = self.get_beta_and_dot_beta(alpha, dot_alpha)
        
        output = {
            "alpha": alpha,
            "beta": beta,
        }
        
        if return_derivs:
            output.update({
                "dot_alpha": dot_alpha,
                "dot_beta": dot_beta,
            })
            
        return output

class BMMBayesianTrajectory(BayesianTrajectory):
    """Bayesian trajectory optimization with BMM-CDF parameterization"""
    
    def __init__(self, n_components=3, constraint_type="spherical", eps=1e-5, grid_size=200):
        super().__init__(n_components, constraint_type, eps, grid_size)
        
        # BMM specific parameters
        self.log_a = nn.Parameter(torch.ones(n_components) * np.log(2.0))
        self.log_b = nn.Parameter(torch.ones(n_components) * np.log(2.0))
        
        # BMM prior parameters
        self.register_buffer("prior_log_a", torch.ones(n_components) * np.log(2.0))
        self.register_buffer("prior_log_b", torch.ones(n_components) * np.log(2.0))
        
        # Parameter constraints
        self.a_min = 0.1
        self.a_max = 10.0
        self.b_min = 0.1
        self.b_max = 10.0

    def get_params(self, mode="posterior"):
        """Convert unconstrained parameters to constrained ones"""
        if mode == "posterior":
            weights = F.softmax(self.logits, dim=-1)
            a = F.softplus(self.log_a) + self.eps
            b = F.softplus(self.log_b) + self.eps
            
            # Apply additional constraints
            a = torch.clamp(a, self.a_min, self.a_max)
            b = torch.clamp(b, self.b_min, self.b_max)
            
            return weights, a, b
        else:
            weights = F.softmax(self.prior_logits, dim=-1)
            a = F.softplus(self.prior_log_a) + self.eps
            b = F.softplus(self.prior_log_b) + self.eps
            
            # Apply additional constraints
            a = torch.clamp(a, self.a_min, self.a_max)
            b = torch.clamp(b, self.b_min, self.b_max)
            
            return weights, a, b
    
    def distribution_cdf(self, t, weights, a, b):
        """Compute the CDF of the BMM at time t using Beta distributions"""
        batch_size = t.shape[0]
        component_cdfs = torch.zeros(batch_size, self.n_components, device=t.device)
        
        for i in range(self.n_components):
            beta_dist = Beta(a[i], b[i])
            component_cdfs[:, i] = beta_dist.cdf(t.squeeze())
        
        cdf_vals = (weights * component_cdfs).sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        return cdf_vals
    
    def distribution_pdf(self, t, weights, a, b):
        """Compute the PDF of the BMM at time t using Beta distributions"""
        batch_size = t.shape[0]
        component_pdfs = torch.zeros(batch_size, self.n_components, device=t.device)
        
        for i in range(self.n_components):
            beta_dist = Beta(a[i], b[i])
            component_pdfs[:, i] = torch.exp(beta_dist.log_prob(t.squeeze()))
        
        pdf_vals = (weights * component_pdfs).sum(dim=1, keepdim=True)
        return pdf_vals

class KMMBayesianTrajectory(BayesianTrajectory):
    """Bayesian trajectory optimization with KMM-CDF parameterization"""
    
    def __init__(self, n_components=3, constraint_type="spherical", eps=1e-5, grid_size=200):
        super().__init__(n_components, constraint_type, eps, grid_size)
        
        # KMM specific parameters
        centers = torch.linspace(0.1, 0.9, n_components)
        a_init = torch.where(centers < 0.5, 
                            torch.tensor(1.5) + 0.5 * torch.arange(n_components),
                            torch.tensor(3.0) + 2.0 * torch.arange(n_components))
        
        b_init = torch.where(centers < 0.5,
                            torch.tensor(3.0) + 2.0 * (n_components - torch.arange(n_components)),
                            torch.tensor(1.5) + 0.5 * (n_components - torch.arange(n_components)))
        
        self.log_a = nn.Parameter(torch.log(a_init))
        self.log_b = nn.Parameter(torch.log(b_init))
        
        # KMM prior parameters
        self.register_buffer("prior_log_a", torch.log(a_init))
        self.register_buffer("prior_log_b", torch.log(b_init))
        
        # Parameter constraints
        self.a_min = 0.1
        self.a_max = 10.0
        self.b_min = 0.1
        self.b_max = 10.0
        
    def get_params(self, mode="posterior"):
        """Convert unconstrained parameters to constrained ones"""
        if mode == "posterior":
            weights = F.softmax(self.logits, dim=-1)
            a = F.softplus(self.log_a)
            b = F.softplus(self.log_b)
            
            # Apply additional constraints
            a = torch.clamp(a, self.a_min, self.a_max)
            b = torch.clamp(b, self.b_min, self.b_max)
            return weights, a, b
        else:
            weights = F.softmax(self.prior_logits, dim=-1)
            a = F.softplus(self.prior_log_a)
            b = F.softplus(self.prior_log_b)
            return weights, a, b
        
    def stable_pow(self, x, y):
        return torch.exp(y * torch.log1p(x - 1))   # x^y
    
    def distribution_cdf(self, t, weights, a, b):
        """Compute the CDF of the KMM at time t"""
        t = t.clamp(self.eps, 1 - self.eps)
        
        # Kumaraswamy CDF: F(t) = 1 - (1 - t^a)^b
        t_a = self.stable_pow(t, a)
        component_cdfs = 1 - self.stable_pow(1 - t_a, b)
        cdf_vals = (weights * component_cdfs).sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        
        return cdf_vals
    
    def distribution_pdf(self, t, weights, a, b):
        """Compute the PDF of the KMM at time t"""
        t = t.clamp(self.eps, 1 - self.eps)
        
        # Kumaraswamy PDF: f(t) = a*b*t^(a-1)*(1-t^a)^(b-1)
        t_a_minus_1 = self.stable_pow(t, a - 1)
        t_a = self.stable_pow(t, a)
        component_pdfs = a * b * t_a_minus_1 * self.stable_pow(1 - t_a, b - 1)
        pdf_vals = (weights * component_pdfs).sum(dim=1, keepdim=True)
        
        return pdf_vals
    
class ConcatSquashExtendLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatSquashExtendLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper_gate(t.view(-1, 1))) \
            + self._hyper_bias(t.view(-1, 1))
            
def unsqueeze(input, upscale_factor=2):
    '''
    [:, C*r^2, H, W] -> [:, C, H*r, W*r]
    '''
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels = in_channels // (upscale_factor**2)

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor

    input_view = input.contiguous().view(batch_size, out_channels, upscale_factor, upscale_factor, in_height, in_width)

    output = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    return output.view(batch_size, out_channels, out_height, out_width)

def squeeze(input, downscale_factor=2):
    '''
    [:, C, H*r, W*r] -> [:, C*r^2, H, W]
    '''
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels = in_channels * (downscale_factor**2)

    out_height = in_height // downscale_factor
    out_width = in_width // downscale_factor

    input_view = input.contiguous().view(
        batch_size, in_channels, out_height, downscale_factor, out_width, downscale_factor
    )

    output = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return output.view(batch_size, out_channels, out_height, out_width)

class JointScoreModel(nn.Module):
    def __init__(
        self, hidden_dims, input_shape, sde=None, strides=None, conv=False, derivative_type="data_derivative", layer_type="concat", nonlinearity="softplus", num_squeeze=0, args=None
    ):
        super().__init__()
        self.num_squeeze = num_squeeze
        self.sde = sde
        self.eps = 1e-5
        self.derivative_type = derivative_type

        strides = [None] * (len(hidden_dims) + 1)
        base_layer = ConcatSquashExtendLinear

        # build layers and add them
        layers = []
        activation_fns = []
        hidden_shape = input_shape
        
        if derivative_type == "data_derivative":
            output_dim = input_shape[0]
        elif derivative_type == "time_derivative":
            output_dim = 1
        elif derivative_type == "joint_derivative":
            output_dim = input_shape[0] + 1
        else:
            raise ValueError(f"derivative_type error : {derivative_type}")
        
        dims = hidden_dims + [output_dim]
        for dim_out, stride in zip(dims, strides):
            if stride is None:
                layer_kwargs = {}
            elif stride == 1:
                layer_kwargs = {"ksize": 3, "stride": 1, "padding": 1, "transpose": False}
            elif stride == 2:
                layer_kwargs = {"ksize": 4, "stride": 2, "padding": 1, "transpose": False}
            elif stride == -2:
                layer_kwargs = {"ksize": 4, "stride": 2, "padding": 1, "transpose": True}
            else:
                raise ValueError('Unsupported stride: {}'.format(stride))

            layer = base_layer(hidden_shape[0], dim_out, **layer_kwargs)
            layers.append(layer)
            activation_fns.append(nn.LeakyReLU())

            hidden_shape = list(copy.copy(hidden_shape))
            hidden_shape[0] = dim_out
            if stride == 2:
                hidden_shape[1], hidden_shape[2] = hidden_shape[1] // 2, hidden_shape[2] // 2
            elif stride == -2:
                hidden_shape[1], hidden_shape[2] = hidden_shape[1] * 2, hidden_shape[2] * 2

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])
        self.nfe = 0
        
    def reset_states(self):
        self.nfe = 0

    def forward(self, t, y):
        self.nfe += 1
        dx = y
        # squeeze
        for _ in range(self.num_squeeze):
            dx = squeeze(dx, 2)
        for l, layer in enumerate(self.layers):
            dx = layer(t, dx)
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        # unsqueeze
        for _ in range(self.num_squeeze):
            dx = unsqueeze(dx, 2)
            
        if self.derivative_type == "joint_derivative":
            dx, dt = torch.split(dx, [y.shape[1], dx.shape[1] - y.shape[1]], dim=1)
            return dx, dt
        return dx

class TimeSampler(nn.Module): 
    def __init__(self, weights_fn=None, uniform_eps=1e-5, min_weight=1e-8, 
                 update_freq=100, grid_size=10000, device='cpu', 
                 t_mode="IS", mu=-0.4, sigma=1, aligned_ratio=0.25, 
                 use_interval_annealing=False, annealing_total_steps=50000, annealing_initial_width=0.01,
                 annealing_tangent_ramp_up_start_step=20000, final_tangent_ratio=0.75,):
        """
        Args:
            ...
            aligned_ratio (float): The ratio of samples where t=l when NOT using annealing.
            final_tangent_ratio (float): The final target proportion of samples where l != t.
        """
        super().__init__()
        self.weights_fn = weights_fn
        self.eps = uniform_eps
        self.grid_size = grid_size
        self.min_weight = min_weight 
        self.update_freq = update_freq
        self.device = device
        self.aligned_ratio = aligned_ratio # Used when annealing is OFF
        
        # Interval Annealing parameters
        self.use_interval_annealing = use_interval_annealing
        self.annealing_total_steps = float(annealing_total_steps)
        self.annealing_initial_width = annealing_initial_width
        self.tangent_start_step = annealing_total_steps * 0.2 
        self.final_aligned_ratio = 1.0 - final_tangent_ratio 
        
        self.register_buffer('t_grid', torch.linspace(uniform_eps, 1-uniform_eps, grid_size, device=device).unsqueeze(-1))
        self.register_buffer('weights', torch.ones(grid_size, device=device) / grid_size)
        self.step = 0
        
        if t_mode == "IS" and weights_fn:
            self.sampling_fn = self.importance_sampling
        elif t_mode == "lognorm":
            self.sampling_fn = lambda batch_size, device : self.lognorm(batch_size, mu=mu, sigma=sigma, device=device)
        else:
            self.sampling_fn = self.uniform

    def forward(self):
        pass
        
    def uniform(self, batch_size, device='cpu'):
        return torch.rand(batch_size, 1, device=device) * (1 - 2*self.eps) + self.eps
    
    def lognorm(self, batch_size, mu=-0.4, sigma=1, device='cpu'):
        eta = torch.randn((batch_size, 1), device=device) * sigma + mu
        return torch.sigmoid(eta).clamp(self.eps, 1 - self.eps)
    
    def importance_sampling(self, batch_size, device='cpu'):   
        if self.step % self.update_freq==1:
            self.update_weights()
        idx = torch.multinomial(self.weights, batch_size, replacement=True)
        return self.t_grid[idx].to(device)

    @torch.no_grad()
    def update_weights(self):
        """Recompute sampling probabilities over the grid."""
        w = self.weights_fn(self.t_grid).clamp(min=1e-12).squeeze()
        self.weights.copy_(torch.clamp(w / w.sum(), min=self.min_weight))
            
    def sample_t(self, batch_size, device="cpu"):
        """ Sample a batch of t values of shape [batch_size, 1]."""
        self.step += 1
        return self.sampling_fn(batch_size, device)
        
    def sample_l_t(self, batch_size, device="cpu"):
        self.step += 1
        
        l = self.sampling_fn(batch_size, device)
        t = self.sampling_fn(batch_size, device)
        
        # --- Annealing Logic Branch ---
        if self.use_interval_annealing:
            progress = min(1.0, self.step / self.annealing_total_steps)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            
            current_max_width = self.annealing_initial_width + (1 - cosine_factor) * (1.0 - self.annealing_initial_width)
            current_aligned_ratio = max(self.final_aligned_ratio, 0.8 * (1 - progress))

            delta = (l - t).abs()
            scale_factor = torch.clamp(current_max_width / delta, max=1.0)
            l_constrained = t + (l - t) * scale_factor

            mask = torch.rand(batch_size, 1, device=device) < current_aligned_ratio
            l = torch.where(mask, t, l_constrained)
            
        # --- Non-Annealing (Original) Logic Branch ---
        else:
            mask = torch.rand(batch_size, 1, device=device) < self.aligned_ratio
            l = torch.where(mask, t, l)
        
        # Ensure l <= t for a consistent interval definition.
        l, t = torch.min(l, t), torch.max(l, t)
        
        return l, t

class TestModel(nn.Module):
    def __init__(self, beta_min=0.1, beta_max=20, eps=1e-5, 
                 combination_type="VPSDE", input_dim=1, joint=True, device="cpu", bridge=False):
        super().__init__()
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        
        self.bridge = bridge
        self.gamma_t = 1.
        
        self.eps = eps
        self.combination_type = combination_type  # e.g., "linear", "VPSDE", "cosine"
        input_shape = (input_dim,)
        self.score_model = JointScoreModel(
                hidden_dims=[64, 64, 64],
                input_shape=input_shape,
                derivative_type="joint_derivative" if joint else "time_derivative",
            ).to(device)
            
        self.gpp = True  # Gaussian Process Prior
        if combination_type == "gpp":
            
            self.gaussian_process = KMMBayesianTrajectory(
                n_components=10, 
                constraint_type="spherical", 
                eps=eps, 
                grid_size=200
            ).to(device)
            
        self.step = 0
        
        weight_fn = lambda t: self.get_weights_denoise(t, mean=True)[0]
        self.t_mode = "AIS"
        self.t_sampler = TimeSampler(weights_fn=weight_fn,
                                     t_mode=self.t_mode,
                                     uniform_eps=eps, 
                                     device=device)
    
    def forward(self, samples):
        x0, x1 = samples 
        device = x0.device
        batch_size = x0.size(0)
        dim = x0.numel() // batch_size
        eps = self.eps
        
        t = self.t_sampler.sample_t(batch_size, device=device)
        if torch.cuda.is_available():
            t = t.to(device)
        
        x0 = x0.to(device)
        x1 = x1.to(device)
        t = t.to(device)
        
        x0_norm_sq = x0.flatten(1).pow(2).sum(1, keepdim=True)
        x1_norm_sq = x1.flatten(1).pow(2).sum(1, keepdim=True)
        x0x1_norm_sq = (x0 * x1).flatten(1).sum(1, keepdim=True)
        
        alpha_t, beta_t, d_alpha_dt, d_beta_dt = self.get_coefficients_and_derivatives(t)
        xt = self.marginal_sample(x0, x1, t, alpha_t=alpha_t, beta_t=beta_t)
        score_x, score_t = self.score_model(t, xt)
        
        if self.combination_type == "gpp":
            variance = (2 * dim * d_alpha_dt**2 + x1_norm_sq * d_beta_dt**2) / (alpha_t**2 + 1e-5)
            # variance += d_alpha_dt**2 + d_beta_dt**2
            score_reg = score_t**2
            elbo = self.gaussian_process.elbo(x0, x1, t, variance=variance, score_reg=score_reg)
        
        if self.t_mode == "IS":
            lambda_t_time = torch.ones_like(t, device=device)
            lambda_t_data = stopgrad(2 * dim * d_alpha_dt**2 + x1_norm_sq * d_beta_dt**2) 
        else:
            marginal_var = alpha_t ** 2
            marginal_var_dt = 2 * alpha_t * d_alpha_dt
            norm_term = d_beta_dt**2 * x1_norm_sq
            
            time_score_var = 0.5 * dim * (marginal_var_dt / marginal_var)**2 + norm_term / (marginal_var)
            data_score_var = 1. / (marginal_var)
            
            lambda_t_time = 1. / time_score_var
            lambda_t_data = 1. / data_score_var
        
        # Time score loss
        target_cond_time_score = (- dim * d_alpha_dt + x0x1_norm_sq * d_beta_dt + x0_norm_sq * d_alpha_dt) / (alpha_t + 1e-5)
        time_loss = (score_t - stopgrad(target_cond_time_score))**2 * lambda_t_time 

        # data score loss
        target_cond_data_score = - x0 / (alpha_t + 1e-8)
        diff = score_x - stopgrad(target_cond_data_score)
        ssm_loss = torch.sum(diff ** 2, dim=tuple(range(1, diff.ndim)), keepdim=True) * lambda_t_data
        
        loss = time_loss + ssm_loss
        
        if self.gpp:
            loss += - elbo
            if self.step % 100 == 0:
                print("\n Loss: {:.2f}, time: {:.2f}, ssm: {:.2f}, elbo: {:.2f}".format(
                      loss.detach().mean().item(), 
                      time_loss.detach().mean().item(), 
                      ssm_loss.detach().mean().item(), 
                      elbo.detach().mean().item(),
                      )
                )
        
        self.step += 1
        return loss
            
    def get_weights_denoise(self, t, x0=None, x1=None, x1_norm_sq=None, alpha_t=None, beta_t=None, 
                       d_alpha_dt=None, d_beta_dt=None, combination_type=None, mean=False):
        if alpha_t is None:
            alpha_t, beta_t, d_alpha_dt, d_beta_dt = self.get_coefficients_and_derivatives(t, mean=mean)
        if x1 is None:
            dim = 1.
            x1_norm_sq = 1.
        else:
            batch_size = x1.size(0)
            dim = x1.numel() // batch_size
            if x1_norm_sq is None:
                x1_norm_sq = x1.flatten(1).pow(2).sum(1, keepdim=True)
        return alpha_t**2 / (2 * dim * d_alpha_dt**2 + x1_norm_sq * d_beta_dt**2), alpha_t**2
        # return (2 * dim * d_alpha_dt**2 + x1_norm_sq * d_beta_dt**2 + 1e-5) / (alpha_t**2 + 1e-5), 1 / (alpha_t**2 + 1e-5)
        
    def get_coefficients_and_derivatives(self, t, mean=False):
        combination_type = self.combination_type
        if combination_type in ["gpp"]:
            coefficients = self.gaussian_process.sample_at_time(t, mode="posterior", return_derivs=True, mean=mean)
            alpha_t = coefficients["alpha"]
            beta_t = coefficients["beta"]
            d_alpha_dt = coefficients["dot_alpha"]
            d_beta_dt = coefficients["dot_beta"]
        else:
            alpha_t, beta_t = self.get_alpha_t_beta_t(t, combination_type=combination_type)
            d_alpha_dt, d_beta_dt = self.get_d_alpha_beta_dt(t, alpha_t=alpha_t, beta_t=beta_t, combination_type=combination_type)
        return alpha_t, beta_t, d_alpha_dt, d_beta_dt

    def get_alpha_t_beta_t(self, t, combination_type=None):
        if combination_type is None:
            combination_type = self.combination_type
        if combination_type == "linear":
            alpha_t = 1 - t
            beta_t = t
        elif combination_type == "VPSDE":
            log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
            alpha_t = torch.exp(log_mean_coeff)
            beta_t = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        elif combination_type == "follmer":
            alpha_t = torch.sqrt(1 - t ** 2)
            beta_t = t
        elif combination_type == "cosine":
            s = torch.tensor(0.008, dtype=t.dtype, device=t.device)
            fn_t = torch.cos((t + s) / (1 + s) * math.pi / 2)
            fn_0 = torch.cos(s / (1 + s) * math.pi / 2)
            alpha_t_bar = fn_t / fn_0
            alpha_t = torch.sqrt(alpha_t_bar)
            beta_t = torch.sqrt(1 - alpha_t_bar)
        elif combination_type == "trigonometric":
            x = math.pi * t * 0.5
            alpha_t = torch.cos(x)
            beta_t = torch.sin(x)
        elif combination_type == "gpp":   # Gaussian Process Prior
            coefficients = self.gaussian_process.sample_at_time(t, mode="posterior", return_derivs=True)
            alpha_t = coefficients["alpha"].view(-1, 1)
            beta_t = coefficients["beta"].view(-1, 1)
        else:
            raise ValueError(f"Combination type {self.combination_type} not implemented in InterpXt.")
        return alpha_t, beta_t
      
    def get_d_alpha_beta_dt(self, t, alpha_t=None, beta_t=None, combination_type=None):
        if combination_type is None:
            combination_type = self.combination_type
        if alpha_t is None:
            alpha_t, beta_t = self.get_alpha_t_beta_t(t, combination_type=combination_type)
        if combination_type == "linear":
            d_alpha_dt = -1.
            d_beta_dt = 1.
        elif combination_type == "VPSDE":
            term_common = 0.5 * ((self.beta_1 - self.beta_0) * t + self.beta_0)
            d_alpha_dt = -term_common * alpha_t
            d_beta_dt = alpha_t ** 2 / (beta_t + 1e-6) * term_common
        elif combination_type == "follmer":
            d_alpha_dt = -t / (torch.sqrt(1 - t**2) + 1e-6)
            d_beta_dt = 1.
        elif combination_type == "cosine":
            s = torch.tensor(0.008, dtype=t.dtype, device=t.device)
            const = math.pi * 0.5 / (1 + s)
            theta_t = (t + s) * const
            sin_theta_t = torch.sin(theta_t)
            theta_0 = s * const
            fn_0 = torch.cos(theta_0)
            
            d_alpha_bar_dt = -sin_theta_t * const / fn_0
            d_alpha_dt = 0.5 * d_alpha_bar_dt / alpha_t
            d_beta_dt = -0.5 * d_alpha_bar_dt / beta_t
        elif combination_type == "trigonometric":
            pi_half = math.pi / 2.0
            d_alpha_dt = -pi_half * beta_t
            d_beta_dt = pi_half * alpha_t
        elif combination_type == "gpp":   # Gaussian Process Prior
            coefficients = self.gaussian_process.sample_at_time(t, mode="posterior", return_derivs=True)
            d_alpha_dt = coefficients["dot_alpha"]
            d_beta_dt = coefficients["dot_beta"]
        
        else:
            raise ValueError(f"Combination type {combination_type} not implemented in InterpXt.")
        
        return d_alpha_dt, d_beta_dt
    
    def marginal_sample(self, x0, x1, t, alpha_t=None, beta_t=None):
        if alpha_t is None:    
            alpha_t, beta_t = self.get_alpha_t_beta_t(t)
        xt = alpha_t * x0 + beta_t * x1
        return xt

def get_alpha_t_beta_t(t, combination_type="VPSDE"):
    if combination_type == "linear":
        alpha_t = 1 - t
        beta_t = t
    elif combination_type == "VPSDE":
        beta_0 = 0.1
        beta_1 = 20
        log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
        alpha_t = np.exp(log_mean_coeff)
        beta_t = np.sqrt(np.maximum(1. - np.exp(2. * log_mean_coeff), 1e-8))
    elif combination_type == "cosine":
        s = 0.008
        fn = lambda x: np.cos((x + s) / (1 + s) * np.pi / 2)
        alpha_t_bar = fn(t) / fn(0)
        alpha_t = np.sqrt(alpha_t_bar)
        beta_t = np.sqrt(np.maximum(1 - alpha_t_bar, 1e-8))
    elif combination_type == "follmer":
        alpha_t = np.sqrt(np.maximum(1 - t ** 2, 1e-8))
        beta_t = t
    elif combination_type == "trigonometric":
        x = np.pi * t * 0.5
        alpha_t = np.cos(x)
        beta_t = np.sin(x)
    else:
        raise ValueError(f"Not implemented combination type: {combination_type}")
    return alpha_t, beta_t

def get_d_alpha_beta_dt(t, alpha_t, beta_t, combination_type="VPSDE"):
    if combination_type == "linear":
        dot_alpha_t = -np.ones_like(t)
        dot_beta_t = np.ones_like(t)
    elif combination_type == "VPSDE":
        beta_0 = 0.1
        beta_1 = 20
        term_common = 0.5 * ((beta_1 - beta_0) * t + beta_0)
        dot_alpha_t = -term_common * alpha_t
        dot_beta_t = (alpha_t ** 2 / beta_t) * term_common
    elif combination_type == "cosine":
        s = 0.008
        const = np.pi * 0.5 / (1 + s)
        sin_theta_t = np.sin((t + s) * const)
        fn_0 = np.cos(s * const)
        d_alpha_bar_dt = -sin_theta_t * const / fn_0
        dot_alpha_t = 0.5 * d_alpha_bar_dt / alpha_t
        dot_beta_t = -0.5 * d_alpha_bar_dt / beta_t
    elif combination_type == "follmer":
        dot_alpha_t = -t / alpha_t
        dot_beta_t = np.ones_like(t)
    elif combination_type == "trigonometric":
        pi_half = np.pi / 2.0
        dot_alpha_t = -pi_half * beta_t
        dot_beta_t = pi_half * alpha_t
    else:
        raise ValueError(f"Not implemented combination type: {combination_type}")
    return dot_alpha_t, dot_beta_t

def visualize_path_function(model, num_samples, prior_traj=None, post_traj=None, name="alpha", t_grid=None, sample_path="./"):
    if prior_traj is None:
        prior_traj = model.gaussian_process.sample_trajectories(mode="prior")
        post_traj = model.gaussian_process.sample_trajectories(mode="posterior")
        t_grid = model.gaussian_process.grid.cpu()
    alpha_prior = prior_traj[name].detach().cpu()
    alpha_post = post_traj[name].detach().cpu()

    ylabel = {"alpha": '$\\alpha_t$', 
             "beta": '$\\beta_t$', 
             "gamma": '$\\gamma_t$'}[name]

    plt.figure(figsize=(8, 6))
    
    # Overlay existing-method trajectories
    methods = ['linear', 'VPSDE', 'cosine', 'follmer', 'trigonometric']
    colors = {'linear': 'green', 'VPSDE': 'black', 'cosine': 'orange', 'follmer': 'pink', 'trigonometric': 'purple'}
    labels = {'linear': 'Linear', 'VPSDE': 'VPSDE', 'cosine': 'Cosine', 'follmer': 'Follmer', 'trigonometric': 'Trigonometric'}
    
    # For each method, compute its trajectory and plot
    if name != "gamma":
        for m in methods:
            method_alpha, method_beta = get_alpha_t_beta_t(t_grid, combination_type=m)
            if name == "alpha":
                plt.plot(t_grid, method_alpha, color=colors[m], lw=2, linestyle='-', label=f'{labels[m]}')
            elif name == "beta":
                plt.plot(t_grid, method_beta, color=colors[m], lw=2, linestyle='-', label=f'{labels[m]}')

    # Plot prior/posterior means
    plt.plot(t_grid, alpha_prior, 'b-', lw=2.5, label='KMM (prior, ours)')
    plt.plot(t_grid, alpha_post, 'r-', lw=2.5, label='KMM (posterior, ours)')
    
    plt.title(f'Trajectory Comparison ({ylabel})', fontsize=16)
    plt.xlabel('Time $t$', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    # save_path = os.path.join(sample_path, "path_comparison.png")
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def calculate_simplified_variance(t, alpha_t, beta_t, dot_alpha_t, dot_beta_t, gamma=1.0, epsilon=1e-4, bridge=False):
    """
    Calculates the simplified DDBI variance based on the provided formula.
    Assumes d=1, E[||x0||^2]=1, E[||x1||^2]=1, E[x0^T x1]=1.
    """
    if bridge:
        ############ bridge ###############
        # Noise schedule sigma_t^2 and its derivative
        sigma_t_sq = t * (1 - t) * gamma**2 + (alpha_t**2 + beta_t**2) * epsilon
        dot_sigma_t_sq = (1 - 2 * t) * gamma**2 + (2 * alpha_t * dot_alpha_t + 2 * beta_t * dot_beta_t) * epsilon
        
        # Expectation of the squared norm of the mean field derivative
        # E[|| da*x0 + db*x1 ||^2] = da^2*E[||x0||^2] + db^2*E[||x1||^2] + 2*da*db*E[x0^T x1]
        E_norm_sq = dot_alpha_t**2 + dot_beta_t**2 + 2 * dot_alpha_t * dot_beta_t
        
        # The two terms from the variance formula
        term1 = 0.5 * (dot_sigma_t_sq**2) / (sigma_t_sq**2) # d=1
        term2 = E_norm_sq / sigma_t_sq
        variance = term1 + term2
    else:
        ############ DI ####################
        x1_norm_sq = 1
        dim = 1
        term1 = 2 * dim * dot_alpha_t**2
        term2 = x1_norm_sq * dot_beta_t**2
        variance = (term1 + term2) / (alpha_t**2 + epsilon)
    
    return variance

def plot_variance_profiles(sde, combination_type="VPSDE", bridge=False, gamma=1.0):
    """
    Generates and saves the plot comparing variance profiles of different schedules.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if combination_type == "gpp":
        prior_traj = sde.gaussian_process.sample_trajectories(mode="prior", return_derivs=True)
        post_traj = sde.gaussian_process.sample_trajectories(mode="posterior", return_derivs=True)
        t_grid = sde.gaussian_process.grid.cpu()
        
    else:
        eps = 1e-5  # Time grid, clipped to avoid division by zero in some derivatives (e.g., Follmer)
        t_grid = np.linspace(eps, 1 - eps, 500)
    
    methods = ['linear', 'VPSDE', 'cosine', 'follmer', 'trigonometric']
    colors = {'linear': 'green', 'VPSDE': 'black', 'cosine': 'orange', 'follmer': 'pink', 'trigonometric': 'purple'}
    labels = {'linear': 'Linear', 'VPSDE': 'VPSDE', 'cosine': 'Cosine', 'follmer': 'Follmer', 'trigonometric': 'Trigonometric'}
    
    # Baseline methods
    for method in methods:
        # Get coefficients and their derivatives
        alpha, beta = sde.get_alpha_t_beta_t(t_grid, combination_type=method)
        dot_alpha, dot_beta = sde.get_d_alpha_beta_dt(t_grid, alpha, beta, combination_type=method)

        # Calculate variance
        variance = calculate_simplified_variance(t_grid, alpha, beta, dot_alpha, dot_beta, bridge=bridge, gamma=gamma)
        ax.plot(t_grid, variance, label=labels[method], color=colors[method], linewidth=2.5)
        
    # gpp method
    if combination_type == "gpp":
        alpha = post_traj["alpha"].detach().cpu()
        beta = post_traj["beta"].detach().cpu()
        dot_alpha = post_traj["dot_alpha"].detach().cpu()
        dot_beta = post_traj["dot_beta"].detach().cpu()
        variance = calculate_simplified_variance(t_grid, alpha, beta, dot_alpha, dot_beta, bridge=bridge, gamma=gamma)
        ax.plot(t_grid, variance, 'r-', lw=2.5, label="PAVE-DRE (ours)")

    # --- Plot Customization ---
    ax.set_yscale('log')
    ax.set_title('Path-Dependent Variance of Different Interpolation Schedules', fontsize=18, pad=15)
    ax.set_xlabel('Time $t$', fontsize=14)
    ax.set_ylabel('Log Variance', fontsize=14)
    ax.legend(fontsize=12, loc='upper left')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(-0.05, 1.05)
    ax.axvspan(0.9, 1.02, color='red', alpha=0.1, label='Unstable Region')
    ax.text(0.95, 1e-1, 'Potential Instability', rotation=90, verticalalignment='bottom', color='red', alpha=0.7, fontsize=14)

    plt.tight_layout()
    plt.savefig("variance_comparison_plot.pdf", dpi=300)
    plt.show()

def sample_from_mog(num_samples, gaussians):
    """Sample from mixture of Gaussians"""
    weights = np.array([g[2] for g in gaussians])
    weights /= weights.sum()
    
    samples = []
    for _ in range(num_samples):
        idx = np.random.choice(len(gaussians), p=weights)
        mean, std, _ = gaussians[idx]
        samples.append(np.random.normal(mean, std))
    
    return torch.tensor(samples).view(-1, 1)

def train(model, batch_size=500, device="cpu", epoch_num=1000):
    # Define data distributions
    gaussians1 = [(-7, 1, 0.3), (-4, 1, 0.2), (2, 1, 0.2), (7, 1.5, 0.3)]
    gaussians2 = [(-2, 1.2, 0.5), (1, 1.5, 0.5)]
   
    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.001)
    for epoch in tqdm(range(epoch_num)):
        optimizer.zero_grad()
        
        x0 = sample_from_mog(batch_size, gaussians1)
        x1 = sample_from_mog(batch_size, gaussians2)
        if torch.cuda.is_available():
            x0 = x0.to(device)
            x1 = x1.to(device)

        loss = model((x0, x1)).mean()
        
        if torch.isnan(loss).any():
            print(f"NaN detected in loss at epoch {epoch}")
            break

        loss.backward()
        optimizer.step()

# In[]
# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = TestModel(
        combination_type="gpp", device=device
    )
    
    # Train 
    train(model, batch_size=1000, device=device, epoch_num=1000)
    
    # In[]
    num_samples = 5000
    prior_traj = model.gaussian_process.sample_trajectories(mode="prior")
    post_traj = model.gaussian_process.sample_trajectories(mode="posterior")
    t_grid = model.gaussian_process.grid.cpu().numpy()
    
    # Visualize trajectories
    visualize_path_function(model, num_samples, prior_traj, post_traj, name="alpha", t_grid=t_grid)
    visualize_path_function(model, num_samples, prior_traj, post_traj, name="beta", t_grid=t_grid)
    
    # visualize variance
    plot_variance_profiles(model, combination_type="gpp")
    
    # In[]
    # Sample at specific times
    t_points = torch.tensor([[0.1], [0.3], [0.5], [0.7], [0.7], [0.9]], device=device)  # (5, 1)
    samples = model.gaussian_process.sample_at_time(t_points, mode="posterior")
    print("============================================================")
    print(f"Sampled alpha: {samples['alpha'].squeeze().tolist()}")
    print(f"Sampled beta: {samples['beta'].squeeze().tolist()}")
    
    # # Sample entire path
    # # trajectory_post = model.sample_trajectories(1, mode="posterior")
    # trajectory_post = model.gaussian_process.sample_trajectories(1, mode="posterior")
    # print("============================================================")
    # print(f"entire alpha: {trajectory_post['alpha'].squeeze().tolist()}")
    # print("============================================================")
    # print(f"entire beta: {trajectory_post['beta'].squeeze().tolist()}")
    
