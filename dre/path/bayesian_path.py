import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.func import vmap, jacrev
from torch.distributions import Beta

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
