# -*- coding: utf-8 -*-
import abc
import math
from mpmath.functions.rszeta import z_half
import torch
import numpy as np
import ot as pot
from functools import partial
from typing import Union
import torch.nn.functional as F

from . import bayesian_path as BayesianPath  # noqa: F401

def logit_transform(image, lambd=1e-6):
    image = lambd + (1 - 2 * lambd) * image
    image = torch.log(image) - torch.log1p(-image)
    ldj = F.softplus(image) + F.softplus(-image) + np.log(1 - 2 * lambd)
    ldj = ldj.view(image.size(0), -1)  # (batch,)
    return image, ldj


class BasePath(abc.ABC, torch.nn.Module):
    """Path abstract class (ODE or SDE). Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct a path.

        Args:
          N: number of discretization time steps.
        """
        super(BasePath, self).__init__()
        abc.ABC.__init__(self)  # ABC
        self.N = N

    def forward(self, x, t):
        pass

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the path."""
        pass

    def drift_diffusion(self, x, t):
        """Drift and diffusion (or ODE velocity). Returns (drift, diffusion)."""
        pass

    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the path, $p_t(x)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, $p_T(x)$."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
          z: latent code
        Returns:
          log probability density
        """
        pass

    def discretize(self, x, t):
        """Discretize the path in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probability flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
          x: a torch tensor
          t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
          f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.drift_diffusion(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(self, score_fn, probability_flow=False):
        """Create the reverse-time path (ODE or SDE).

        Args:
          score_fn: A time-dependent score-based model that takes x and t and returns the score.
          probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = self.N
        T = self.T
        drift_diffusion_fn = self.drift_diffusion
        discretize_fn = self.discretize

        class RBasePath(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def drift_diffusion(self, x, t):
                """Drift and diffusion for the reverse path."""
                drift, diffusion = drift_diffusion_fn(x, t)
                score = score_fn(x, t)
                # TODO: added this
                if isinstance(score, list) or isinstance(score, tuple):
                    score = score[0]
                drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
                # Set the diffusion function to zero for ODEs.
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t)
                # TODO: added this
                score = score_fn(x, t)
                if isinstance(score, list) or isinstance(score, tuple):
                    score = score[0]
                rev_f = f - G[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RBasePath()

class OTPlanSampler:
    """OTPlanSampler implements sampling coordinates according to an OT plan (wrt squared Euclidean
    cost) with different implementations of the plan calculation."""

    def __init__(
        self,
        method: str,
        reg: float = 0.05,
        reg_m: float = 1.0,
        normalize_cost: bool = False,
        num_threads: Union[int, str] = 1,
        warn: bool = True,
    ) -> None:
        """Initialize the OTPlanSampler class.

        Parameters
        ----------
        method: str
            choose which optimal transport solver you would like to use.
            Currently supported are ["exact", "sinkhorn", "unbalanced",
            "partial"] OT solvers.
        reg: float, optional
            regularization parameter to use for Sinkhorn-based iterative solvers.
        reg_m: float, optional
            regularization weight for unbalanced Sinkhorn-knopp solver.
        normalize_cost: bool, optional
            normalizes the cost matrix so that the maximum cost is 1. Helps
            stabilize Sinkhorn-based solvers. Should not be used in the vast
            majority of cases.
        num_threads: int or str, optional
            number of threads to use for the "exact" OT solver. If "max", uses
            the maximum number of threads.
        warn: bool, optional
            if True, raises a warning if the algorithm does not converge
        """
        # ot_fn should take (a, b, M) as arguments where a, b are marginals and
        # M is a cost matrix
        if method == "exact":
            self.ot_fn = partial(pot.emd, numThreads=num_threads)
        elif method == "sinkhorn":
            self.ot_fn = partial(pot.sinkhorn, reg=reg)
        elif method == "unbalanced":
            self.ot_fn = partial(pot.unbalanced.sinkhorn_knopp_unbalanced, reg=reg, reg_m=reg_m)
        elif method == "partial":
            self.ot_fn = partial(pot.partial.entropic_partial_wasserstein, reg=reg)
        else:
            raise ValueError(f"Unknown method: {method}")
        self.reg = reg
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost
        self.warn = warn

    def get_map(self, x0, x1):
        """Compute the OT plan (wrt squared Euclidean cost) between a source and a target
        minibatch.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the source minibatch

        Returns
        -------
        p : numpy array, shape (bs, bs)
            represents the OT plan between minibatches
        """
        a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        M = torch.cdist(x0, x1) ** 2
        if self.normalize_cost:
            M = M / M.max()  # should not be normalized when using minibatches
        p = self.ot_fn(a, b, M.detach().cpu().numpy())
        if not np.all(np.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x0, x1)
        if np.abs(p.sum()) < 1e-8:
            if self.warn:
                pass
                # warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            p = np.ones_like(p) / p.size
        return p

    def sample_map(self, pi, batch_size, replace=True):
        r"""Draw source and target samples from pi  $(x,z) \sim \pi$

        Parameters
        ----------
        pi : numpy array, shape (bs, bs)
            represents the source minibatch
        batch_size : int
            represents the OT plan between minibatches
        replace : bool
            represents sampling or without replacement from the OT plan

        Returns
        -------
        (i_s, i_j) : tuple of numpy arrays, shape (bs, bs)
            represents the indices of source and target data samples from $\pi$
        """
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(
            pi.shape[0] * pi.shape[1], p=p, size=batch_size, replace=replace
        )
        return np.divmod(choices, pi.shape[1])

    def sample_plan(self, x0, x1, replace=True):
        r"""Compute the OT plan $\pi$ (wrt squared Euclidean cost) between a source and a target
        minibatch and draw source and target samples from pi $(x,z) \sim \pi$

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the source minibatch
        replace : bool
            represents sampling or without replacement from the OT plan

        Returns
        -------
        x0[i] : Tensor, shape (bs, *dim)
            represents the source minibatch drawn from $\pi$
        x1[j] : Tensor, shape (bs, *dim)
            represents the source minibatch drawn from $\pi$
        """
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)
        return x0[i], x1[j]

    def sample_plan_with_labels(self, x0, x1, y0=None, y1=None, replace=True):
        r"""Compute the OT plan $\pi$ (wrt squared Euclidean cost) between a source and a target
        minibatch and draw source and target labeled samples from pi $(x,z) \sim \pi$

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        y0 : Tensor, shape (bs)
            represents the source label minibatch
        y1 : Tensor, shape (bs)
            represents the target label minibatch
        replace : bool
            represents sampling or without replacement from the OT plan

        Returns
        -------
        x0[i] : Tensor, shape (bs, *dim)
            represents the source minibatch drawn from $\pi$
        x1[j] : Tensor, shape (bs, *dim)
            represents the target minibatch drawn from $\pi$
        y0[i] : Tensor, shape (bs, *dim)
            represents the source label minibatch drawn from $\pi$
        y1[j] : Tensor, shape (bs, *dim)
            represents the target label minibatch drawn from $\pi$
        """
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)
        return (
            x0[i],
            x1[j],
            y0[i] if y0 is not None else None,
            y1[j] if y1 is not None else None,
        )

    def sample_trajectory(self, X):
        """Compute the OT trajectories between different sample populations moving from the source
        to the target distribution.

        Parameters
        ----------
        X : Tensor, (bs, times, *dim)
            different populations of samples moving from the source to the target distribution.

        Returns
        -------
        to_return : Tensor, (bs, times, *dim)
            represents the OT sampled trajectories over time.
        """
        times = X.shape[1]
        pis = []
        for t in range(times - 1):
            pis.append(self.get_map(X[:, t], X[:, t + 1]))

        indices = [np.arange(X.shape[0])]
        for pi in pis:
            j = []
            for i in indices[-1]:
                j.append(np.random.choice(pi.shape[1], p=pi[i] / pi[i].sum()))
            indices.append(np.array(j))

        to_return = []
        for t in range(times):
            to_return.append(X[:, t][indices[t]])
        to_return = np.stack(to_return, axis=1)
        return to_return

class InterpXt(BasePath):
    def __init__(self, args=None, N=1000, beta_min=0.1, beta_max=20, data_ndim=2):
        super().__init__(N)
        self.args = args
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N    
        self.data_ndim = data_ndim # 2 for toy and 4 for image
        
        self.sample_noise_std = args.sample_noise_std
        self.bridge = args.bridge
        self.path_type = args.path_type  # e.g., "linear", "VP", "cosine"
        self.gamma_t = args.gamma_t  # in fact is gamma^2
        self.joint = True if self.args.energy else args.joint
        
        self.OT = args.OT
        if self.OT:
            self.ot_sampler = OTPlanSampler(method="sinkhorn")
            if self.path_type != "linear":
                raise ValueError(f"OT is {self.OT} and path_type is {self.path_type}, not linear")
        else:
            self.ot_sampler = None

        self.mvp = args.mvp  # Minimum Variance Path
        if self.mvp:
            self.bayesian_path = BayesianPath.KMMBayesianTrajectory(
                n_components=args.K_gmm, 
                constraint_type=args.constraint_type, 
                eps=args.eps, 
                grid_size=args.grid_size
            )
            
        self.z = 0.
        bridge = self.bridge
        self.set_coefficients_and_derivatives_fns()
        self.set_marginal_mean_fns(bridge)
        self.set_marginal_std_fns(bridge)
        self.set_marginal_var_fns(bridge)
        self.set_marginal_sample_fn(bridge)
        self.set_marginal_derivative_fn(bridge)
        self.set_score_var_fns(bridge)
        self.set_score_target_fns(bridge, joint=self.joint)
        self.set_inner_product_fn()

    @property
    def T(self):
        return 1.0
    
    def set_z(self, z=None):
        self.z = z   # save random noise for bridge case
        
    def set_inner_product_fn(self):
        if self.data_ndim == 2:
            self.inner_product_fn = lambda x, y: (x * y).sum(1, keepdim=True)
        elif self.data_ndim == 4:
            self.inner_product_fn = lambda x, y: (x * y).sum(dim=(1, 2, 3), keepdim=True)
        
    def get_coefficients_and_derivatives_fns(self, path_type):
        if path_type == "linear":
            def coefficients_fn(t):
                return 1 - t, t   # alpha_t, beta_t
            def derivatives_fn(t, alpha_t=None, beta_t=None):
                return - torch.ones_like(t), torch.ones_like(t)
            
        elif path_type == "VP":
            def coefficients_fn(t):
                log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
                alpha_t = torch.exp(log_mean_coeff)
                beta_t = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
                return alpha_t, beta_t 
            def derivatives_fn(t, alpha_t=None, beta_t=None):
                term_common = 0.5 * ((self.beta_1 - self.beta_0) * t + self.beta_0)
                d_alpha_dt = -term_common * alpha_t
                d_beta_dt = alpha_t ** 2 / (beta_t + 1e-6) * term_common
                return d_alpha_dt, d_beta_dt
            
        elif path_type == "follmer":
            def coefficients_fn(t):
                return torch.sqrt(1 - t ** 2), t
            def derivatives_fn(t, alpha_t=None, beta_t=None):
                return -t / (torch.sqrt(1 - t**2) + 1e-6), torch.ones_like(t)
            
        elif path_type == "cosine":
            def coefficients_fn(t):
                s = torch.tensor(0.008, dtype=t.dtype, device=t.device)
                fn_t = torch.cos((t + s) / (1 + s) * math.pi / 2)
                fn_0 = torch.cos(s / (1 + s) * math.pi / 2)
                alpha_t_bar = fn_t / fn_0
                alpha_t = torch.sqrt(alpha_t_bar)
                beta_t = torch.sqrt(1 - alpha_t_bar)
                return alpha_t, beta_t
            def derivatives_fn(t, alpha_t=None, beta_t=None):
                s = torch.tensor(0.008, dtype=t.dtype, device=t.device)
                const = math.pi * 0.5 / (1 + s)
                sin_theta_t = torch.sin((t + s) * const)
                fn_0 = torch.cos(s * const)
                
                d_alpha_bar_dt = -sin_theta_t * const / fn_0
                d_alpha_dt = 0.5 * d_alpha_bar_dt / alpha_t
                d_beta_dt = -0.5 * d_alpha_bar_dt / beta_t
                return d_alpha_dt, d_beta_dt
            
        elif path_type == "trigonometric":
            def coefficients_fn(t):
                x = math.pi * t * 0.5
                alpha_t = torch.cos(x)
                beta_t = torch.sin(x)
                return alpha_t, beta_t
            def derivatives_fn(t, alpha_t=None, beta_t=None):
                pi_half = math.pi / 2.0
                d_alpha_dt = -pi_half * beta_t
                d_beta_dt = pi_half * alpha_t
                return d_alpha_dt, d_beta_dt
            
        elif path_type == "mvp":   # Gaussian Process Prior
            def coefficients_fn(t):
                coefficients = self.bayesian_path.sample_at_time(t, mode="posterior", return_derivs=True)
                alpha_t = coefficients["alpha"]
                beta_t = coefficients["beta"]
                return alpha_t, beta_t
            def derivatives_fn(t, alpha_t=None, beta_t=None):
                coefficients = self.bayesian_path.sample_at_time(t, mode="posterior", return_derivs=True)
                d_alpha_dt = coefficients["dot_alpha"]
                d_beta_dt = coefficients["dot_beta"]
                return d_alpha_dt, d_beta_dt
            
        else:
            raise ValueError(f"Path type {self.path_type} not implemented in InterpXt.")
        
        return coefficients_fn, derivatives_fn

    def set_coefficients_and_derivatives_fns(self):
        # for interpolant: xt = alpha_t x0 + beta_t x1
        # coefficients: alpha_t and beta_t; derivatives: d_alpha_dt and d_beta_dt
        coefficients_fn, derivatives_fn = self.get_coefficients_and_derivatives_fns(path_type=self.path_type)
        self.coefficients_fn = coefficients_fn
        self.derivatives_fn = derivatives_fn
        
        if self.path_type == "mvp":
            def coefficients_and_derivatives_fns(t):
                coefficients = self.bayesian_path.sample_at_time(t, mode="posterior", return_derivs=True)
                alpha_t = coefficients["alpha"]
                beta_t = coefficients["beta"]
                d_alpha_dt = coefficients["dot_alpha"]
                d_beta_dt = coefficients["dot_beta"]
                return alpha_t, beta_t, d_alpha_dt, d_beta_dt
            
        else:
            def coefficients_and_derivatives_fns(t):
                alpha_t, beta_t = self.coefficients_fn(t)
                d_alpha_dt, d_beta_dt = self.derivatives_fn(t, alpha_t=alpha_t, beta_t=beta_t)
                return alpha_t, beta_t, d_alpha_dt, d_beta_dt
        self.coefficients_and_derivatives_fns = coefficients_and_derivatives_fns
    
    def set_marginal_mean_fns(self, bridge=False):
        # default: xt = alpha_t x0 + beta_t x1
        # default: p(xt | x1) = N(x; beta_t x1, alpha_t^2 I)
        # bridge: xt = alpha_t x0 + beta_t x1 + \sqrt{gamma^2 t(1-t) + (alpha_t**2 + beta_t**2) * epsilon}z, z \sim N(0, I)
        # p(xt | x0, x1) = N(x; alpha_t x0 + beta_t x1, (gamma^2 t(1-t) + (alpha_t**2 + beta_t**2) * epsilon)I)
        # marginal_mean : marginal mean of the transition kernel
        def default_marginal_mean_fn(t, x0=None, x1=None, alpha_t=None, beta_t=None):
            if alpha_t is None:
                alpha_t, beta_t = self.get_alpha_t_beta_t(t)
            return beta_t * x1
        def default_marginal_mean_dt_fn(t, x0=None, x1=None, alpha_t=None, beta_t=None, d_alpha_dt=None, d_beta_dt=None):
            if d_alpha_dt is None:
                d_alpha_dt, d_beta_dt = self.derivatives_fn(t, alpha_t, beta_t)
            return d_beta_dt * x1
        def bridge_marginal_mean_fn(t, x0=None, x1=None, alpha_t=None, beta_t=None):
            if alpha_t is None:
                alpha_t, beta_t = self.get_alpha_t_beta_t(t)
            return alpha_t * x0 + beta_t * x1
        def bridge_marginal_mean_dt_fn(t, x0=None, x1=None, alpha_t=None, beta_t=None, d_alpha_dt=None, d_beta_dt=None):
            if d_alpha_dt is None:
                d_alpha_dt, d_beta_dt = self.derivatives_fn(t, alpha_t, beta_t)
            return d_alpha_dt * x0 + d_beta_dt * x1
        if bridge:
            self.marginal_mean_fn = bridge_marginal_mean_fn
            self.marginal_mean_dt_fn = bridge_marginal_mean_dt_fn
        else:
            self.marginal_mean_fn = default_marginal_mean_fn
            self.marginal_mean_dt_fn = default_marginal_mean_dt_fn
    
    def set_marginal_std_fns(self, bridge=False):
        def default_marginal_std_fn(t, alpha_t=None, beta_t=None, d_alpha_dt=None, d_beta_dt=None):
            return torch.sqrt(self.marginal_var_fn(t, alpha_t, beta_t))
        def default_marginal_std_dt_fn(t, alpha_t=None, beta_t=None, d_alpha_dt=None, d_beta_dt=None):
            # std(x)' = (sqrt(var(x)))' = 0.5 * var(x)' / std(x)
            return 0.5 * self.marginal_var_dt_fn(t, alpha_t, beta_t, d_alpha_dt, d_beta_dt) / self.marginal_std_fn(t, alpha_t, beta_t, d_alpha_dt, d_beta_dt)
        self.marginal_std_fn = default_marginal_std_fn
        self.marginal_std_dt_fn = default_marginal_std_dt_fn
        
    def set_marginal_var_fns(self, bridge=False):
        # default: xt = alpha_t x0 + beta_t x1
        # default: p(xt | x1) = N(x; beta_t x1, alpha_t^2 I)
        # bridge: xt = alpha_t x0 + beta_t x1 + \sqrt{gamma^2 t(1-t) + (alpha_t**2 + beta_t**2) * epsilon}z, z \sim N(0, I)
        # p(xt | x0, x1) = N(x; alpha_t x0 + beta_t x1, (gamma^2 t(1-t) + (alpha_t**2 + beta_t**2) * epsilon)I)
        # marginal_var : marginal variance of the transition kernel
        def default_marginal_var_fn(t, alpha_t=None, beta_t=None):
            if alpha_t is None:
                alpha_t, _ = self.get_alpha_t_beta_t(t)
            return alpha_t ** 2
        def default_marginal_var_dt_fn(t, alpha_t=None, beta_t=None, d_alpha_dt=None, d_beta_dt=None):
            if alpha_t is None or d_alpha_dt is None:
                alpha_t, beta_t, d_alpha_dt, d_beta_dt = self.get_coefficients_and_derivatives(t)
            return 2 * alpha_t * d_alpha_dt
        def bridge_marginal_var_fn(t, alpha_t=None, beta_t=None):
            if alpha_t is None:
                alpha_t, beta_t = self.get_alpha_t_beta_t(t)
            t = t.reshape(-1, *[1] * (alpha_t.dim() - 1))
            return self.gamma_t * t * (1 - t) + (alpha_t**2 + beta_t**2) * self.sample_noise_std
        def bridge_marginal_var_dt_fn(t, alpha_t=None, beta_t=None, d_alpha_dt=None, d_beta_dt=None):
            if alpha_t is None or d_alpha_dt is None:
                alpha_t, beta_t, d_alpha_dt, d_beta_dt = self.get_coefficients_and_derivatives(t)
            t = t.reshape(-1, *[1] * (alpha_t.dim() - 1))
            return self.gamma_t * (1. - 2 * t) + 2 * (alpha_t * d_alpha_dt + beta_t * d_beta_dt) * self.sample_noise_std
        if bridge:
            self.marginal_var_fn = bridge_marginal_var_fn
            self.marginal_var_dt_fn = bridge_marginal_var_dt_fn
        else:
            self.marginal_var_fn = default_marginal_var_fn
            self.marginal_var_dt_fn = default_marginal_var_dt_fn

    def set_marginal_sample_fn(self, bridge=False):
        def default_marginal_sample_fn(x0, x1, t, alpha_t, beta_t):
            return alpha_t * x0 + beta_t * x1
        def bridge_marginal_sample_fn(x0, x1, t, alpha_t, beta_t):
            bridge_std = self.marginal_std_fn(t, alpha_t=alpha_t, beta_t=beta_t)
            z = torch.randn_like(x0).to(x0)
            self.set_z(z)
            return alpha_t * x0 + beta_t * x1 + bridge_std * z
        self.marginal_sample_fn = bridge_marginal_sample_fn if bridge else default_marginal_sample_fn
    
    def set_marginal_derivative_fn(self, bridge=False):
        # get d_xt_dt, the derivative of xt
        def default_marginal_derivative_fn(x0, x1, t, alpha_t, beta_t, d_alpha_dt, d_beta_dt):
            return d_alpha_dt * x0 + d_beta_dt * x1
        def bridge_marginal_derivative_fn(x0, x1, t, alpha_t, beta_t, d_alpha_dt, d_beta_dt):
            d_mu_dt = self.marginal_mean_dt_fn(t, x0=x0, x1=x1, alpha_t=alpha_t, beta_t=beta_t, d_alpha_dt=d_alpha_dt, d_beta_dt=d_beta_dt)
            return d_mu_dt + self.marginal_std_dt_fn(t, alpha_t, beta_t, d_alpha_dt, d_beta_dt) * self.z
        self.marginal_derivative_fn = bridge_marginal_derivative_fn if bridge else default_marginal_derivative_fn
    
    def set_score_var_fns(self, bridge=False):
        def default_score_var_fn(t, x0=None, x1=None, x0_norm_sq=1., x1_norm_sq=1., x0x1_norm_sq=1., 
                            alpha_t=None, beta_t=None, d_alpha_dt=None, d_beta_dt=None, path_type=None):
            if alpha_t is None:
                alpha_t, beta_t, d_alpha_dt, d_beta_dt = self.get_coefficients_and_derivatives(t)
            if x1 is None:
                dim = 1.
                x1_norm_sq = 1.
            else:
                dim = x1.numel() // x1.size(0)
                if x1_norm_sq is None:
                    x1_norm_sq = self.inner_product(x1, x1)
            norm_term = d_beta_dt**2 * x1_norm_sq
            
            marginal_var = self.marginal_var_fn(t, alpha_t=alpha_t, beta_t=beta_t) + 1e-5
            marginal_var_dt = self.marginal_var_dt_fn(t, alpha_t=alpha_t, beta_t=beta_t, d_alpha_dt=d_alpha_dt, d_beta_dt=d_beta_dt)
            time_score_var = 0.5 * dim * (marginal_var_dt / marginal_var)**2 + norm_term / (marginal_var)
            data_score_var = 1. / (marginal_var)
            return time_score_var, data_score_var  
        
        def bridge_score_var_fn(t, x0=None, x1=None, x0_norm_sq=None, x1_norm_sq=None, x0x1_norm_sq=None, 
                            alpha_t=None, beta_t=None, d_alpha_dt=None, d_beta_dt=None, path_type=None):
            if alpha_t is None:
                alpha_t, beta_t, d_alpha_dt, d_beta_dt = self.get_coefficients_and_derivatives(t)
            if x1 is None:
                dim = 1.
                x0_norm_sq = x1_norm_sq = x0x1_norm_sq = 1.
            else:
                dim = x1.numel() // x1.size(0)
                if x1_norm_sq is None:
                    x0_norm_sq = self.inner_product(x0, x0)
                    x1_norm_sq = self.inner_product(x1, x1)
                    x0x1_norm_sq = self.inner_product(x0, x1)
            norm_term = d_alpha_dt**2 * x0_norm_sq + d_beta_dt**2 * x1_norm_sq + 2 * torch.abs(d_alpha_dt*d_beta_dt) * x0x1_norm_sq
            
            marginal_var = self.marginal_var_fn(t, alpha_t=alpha_t, beta_t=beta_t) + 1e-5
            marginal_var_dt = self.marginal_var_dt_fn(t, alpha_t=alpha_t, beta_t=beta_t, d_alpha_dt=d_alpha_dt, d_beta_dt=d_beta_dt)
            time_score_var = 0.5 * dim * (marginal_var_dt / marginal_var)**2 + norm_term / (marginal_var)
            data_score_var = 1. / (marginal_var)
            return time_score_var, data_score_var   # [BS, 1]
        self.score_var_fn = default_score_var_fn if not bridge else bridge_score_var_fn
        
    def set_score_target_fns(self, bridge=False, joint=False):
        # Target time and data score
        def default_time_score_target(t, x0=None, x1=None, x0_norm_sq=None, x1_norm_sq=None, x0x1_norm_sq=None, 
                                      alpha_t=None, beta_t=None, d_alpha_dt=None, d_beta_dt=None, dim=None):
            return (- dim * d_alpha_dt + x0x1_norm_sq * d_beta_dt + x0_norm_sq * d_alpha_dt) / (alpha_t + 1e-5)
        def default_joint_score_target(t, x0=None, x1=None, x0_norm_sq=None, x1_norm_sq=None, x0x1_norm_sq=None, 
                                      alpha_t=None, beta_t=None, d_alpha_dt=None, d_beta_dt=None, dim=None):
            time_score_target = default_time_score_target(t, x0, x1, x0_norm_sq, x1_norm_sq, x0x1_norm_sq, 
                                      alpha_t, beta_t, d_alpha_dt, d_beta_dt, dim)
            data_score_target = - x0 / (alpha_t + 1e-5)
            return time_score_target, data_score_target
        def bridge_time_score_target(t, x0=None, x1=None, x0_norm_sq=None, x1_norm_sq=None, x0x1_norm_sq=None, 
                                      alpha_t=None, beta_t=None, d_alpha_dt=None, d_beta_dt=None, dim=None):
            if x0_norm_sq.squeeze().mean() == 1.:
                mu_dt_z_norm_sq = d_alpha_dt + d_beta_dt
                z_norm_sq = 1.
                dim = 1.
            else:
                z = self.z
                mu_dt = self.marginal_mean_dt_fn(t, x0, x1, alpha_t, beta_t, d_alpha_dt, d_beta_dt)   # d_alpha_dt * x0 + d_beta_dt * x1
                mu_dt_z_norm_sq = self.inner_product(mu_dt, z)
                z_norm_sq = self.inner_product(z, z)
            var = self.marginal_var_fn(t, alpha_t, beta_t)   
            var_dt = self.marginal_var_dt_fn(t, alpha_t, beta_t, d_alpha_dt, d_beta_dt)   
            common = var_dt / (var + 1e-5)
            return -0.5 * dim * common + mu_dt_z_norm_sq / ((var+1e-5).sqrt()+1e-6) + 0.5 * common * z_norm_sq
        def bridge_joint_score_target(t, x0=None, x1=None, x0_norm_sq=None, x1_norm_sq=None, x0x1_norm_sq=None, 
                                      alpha_t=None, beta_t=None, d_alpha_dt=None, d_beta_dt=None, dim=None):
            time_score_target = bridge_time_score_target(t, x0, x1, x0_norm_sq, x1_norm_sq, x0x1_norm_sq, 
                                      alpha_t, beta_t, d_alpha_dt, d_beta_dt, dim)
            data_score_target = - self.z / (self.marginal_std_fn(t, alpha_t, beta_t, d_alpha_dt, d_beta_dt) + 1e-5)
            return time_score_target, data_score_target
        
        if bridge:
            self.score_target_fn = bridge_joint_score_target if joint else bridge_time_score_target
        else:
            self.score_target_fn = default_joint_score_target if joint else default_time_score_target
    
    ###########################################################################################
    def get_alpha_t_beta_t(self, t, path_type=None):
        if path_type is None:
            return self.coefficients_fn(t)
        else:
            coefficients_fn, derivatives_fn = self.get_coefficients_and_derivatives_fns(path_type=path_type)
            return coefficients_fn(t)
      
    def get_d_alpha_beta_dt(self, t, alpha_t=None, beta_t=None):
        return self.derivatives_fn(t, alpha_t=alpha_t, beta_t=beta_t) #d_alpha_dt, d_beta_dt
    
    def get_coefficients_and_derivatives(self, t):
        return self.coefficients_and_derivatives_fns(t)

    def total_scale_squared(self, t, v0, v1):
        """Path-aware total effective scale s_t^2 for EDM input preconditioning.

        Default path (x_t = alpha_t x0 + beta_t x1):
            s_t^2 = alpha_t^2 * v0 + beta_t^2 * v1
        Bridge path (with additive noise), matching marginal_var_fn and marginal_sample:
            s_t^2 = alpha_t^2 * v0 + beta_t^2 * v1 + gamma^2 * t * (1-t) + (alpha_t^2 + beta_t^2) * epsilon

        Units (must match set_marginal_var_fns / marginal_sample):
        - self.gamma_t stores gamma^2 (variance scale of the t(1-t) term).
        - self.sample_noise_std is used as epsilon (variance) in (alpha_t^2+beta_t^2)*epsilon.
          If it were ever supplied as a standard deviation, the term would be (a2+b2)*sample_noise_std**2.

        v0, v1: scalar or per-channel tensor (C,); correspond to endpoint 0 (alpha_t) and 1 (beta_t).
        Returns: (B, 1) if v0,v1 scalar; (B, C) if per-channel, for broadcasting with x_t.
        """
        alpha_t, beta_t = self.get_alpha_t_beta_t(t)
        alpha_t = alpha_t.reshape(-1, 1)
        beta_t = beta_t.reshape(-1, 1)
        a2 = alpha_t ** 2
        b2 = beta_t ** 2
        v0 = torch.as_tensor(v0, device=t.device, dtype=t.dtype)
        v1 = torch.as_tensor(v1, device=t.device, dtype=t.dtype)
        if v0.dim() == 0 and v1.dim() == 0:
            s_t2 = a2 * v0 + b2 * v1
        else:
            if v0.dim() == 0:
                v0 = v0.unsqueeze(0)
            if v1.dim() == 0:
                v1 = v1.unsqueeze(0)
            v0 = v0.reshape(1, -1)
            v1 = v1.reshape(1, -1)
            s_t2 = a2 * v0 + b2 * v1
        if self.bridge:
            t_flat = t.reshape(-1, 1)
            s_t2 = s_t2 + self.gamma_t * t_flat * (1.0 - t_flat) + (a2 + b2) * self.sample_noise_std
        return s_t2

    def total_scale_squared_t_only(self, t):
        """t-only path-based scale s_t^2 for EDM input preconditioning (no endpoint statistics).

        Default path (x_t = alpha_t x0 + beta_t x1):
            s_t^2 = alpha_t^2 + beta_t^2
        Bridge path (with additive noise), same units as marginal_var_fn:
            s_t^2 = alpha_t^2 + beta_t^2 + gamma^2 t(1-t) + (alpha_t^2 + beta_t^2) * epsilon
        Uses self.gamma_t as gamma^2 and self.sample_noise_std as epsilon (variance).
        Returns: (B, 1) for broadcasting with x_t.
        """
        alpha_t, beta_t = self.get_alpha_t_beta_t(t)
        alpha_t = alpha_t.reshape(-1, 1)
        beta_t = beta_t.reshape(-1, 1)
        a2 = alpha_t ** 2
        b2 = beta_t ** 2
        s_t2 = a2 + b2
        if self.bridge:
            t_flat = t.reshape(-1, 1)
            s_t2 = s_t2 + self.gamma_t * t_flat * (1.0 - t_flat) + (a2 + b2) * self.sample_noise_std
        return s_t2

    def get_weights_denoise(self, t, x0=None, x1=None, x0_norm_sq=None, x1_norm_sq=None, x0x1_norm_sq=None, 
                            alpha_t=None, beta_t=None, d_alpha_dt=None, d_beta_dt=None, path_type=None):
        # weight is inversely proportional to variance, only used for variance-based importance sampling
        time_score_var, data_score_var = self.score_var_fn(t, x0=x0, x1=x1, x0_norm_sq=x0_norm_sq, x1_norm_sq=x1_norm_sq, x0x1_norm_sq=x0x1_norm_sq, 
                            alpha_t=alpha_t, beta_t=beta_t, d_alpha_dt=d_alpha_dt, d_beta_dt=d_beta_dt, path_type=path_type)
        return 1. / time_score_var.view(-1, 1), 1. / data_score_var.view(-1, 1)
    
    def get_score_var(self, t, x0=None, x1=None, x0_norm_sq=None, x1_norm_sq=None, x0x1_norm_sq=None, 
                            alpha_t=None, beta_t=None, d_alpha_dt=None, d_beta_dt=None, path_type=None):
        return self.score_var_fn(t, x0, x1, x0_norm_sq, x1_norm_sq, x0x1_norm_sq, alpha_t, beta_t, d_alpha_dt, d_beta_dt, path_type)

    def get_score_target(self, t, x0=None, x1=None, x0_norm_sq=None, x1_norm_sq=None, x0x1_norm_sq=None, 
                               alpha_t=None, beta_t=None, d_alpha_dt=None, d_beta_dt=None, dim=None):
        return self.score_target_fn(t, x0, x1, x0_norm_sq, x1_norm_sq, x0x1_norm_sq, alpha_t, beta_t, d_alpha_dt, d_beta_dt, dim)
    
    def marginal_sample(self, x0, x1, t, alpha_t=None, beta_t=None):
        if self.OT:
            x0, x1 = self.ot_sampler.sample_plan(x0, x1, replace=True)
        if alpha_t is None:    
            alpha_t, beta_t = self.get_alpha_t_beta_t(t)
        return self.marginal_sample_fn(x0, x1, t, alpha_t, beta_t)
    
    def get_marginal_derivative(self, x0, x1, t, alpha_t=None, beta_t=None, d_alpha_dt=None, d_beta_dt=None):
        if alpha_t is None:    
            alpha_t, beta_t = self.get_alpha_t_beta_t(t)
        if d_alpha_dt is None or d_beta_dt is None:
            d_alpha_dt, d_beta_dt = self.get_d_alpha_beta_dt(t, alpha_t, beta_t)
        return self.marginal_derivative_fn(x0, x1, t, alpha_t, beta_t, d_alpha_dt, d_beta_dt)

    def inner_product(self, x, y):
        assert x.shape == y.shape, "Input tensor shapes must match."
        return self.inner_product_fn(x, y)
    
    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        # Compute the dimension per sample (N = product of all dimensions except batch)
        N = z.numel() // z.size(0)
        # Compute squared sum for each sample in the batch
        squared_sum = torch.sum(z ** 2, dim=tuple(range(1, z.dim())))
        return -0.5 * N * torch.log(2 * torch.tensor(torch.pi, device=z.device, dtype=z.dtype)) - 0.5 * squared_sum

    def drift_diffusion(self, x, t):
        raise NotImplementedError("drift_diffusion must be implemented in subclass")
    
    def discretize(self, x, t):
        raise NotImplementedError("discretize method must be implemented in subclass")
    
class ToyInterpXt(InterpXt):
    def __init__(self, args=None, N=1000, beta_min=0.1, beta_max=20, data_ndim=2):
        super().__init__(args=args, N=N, beta_min=beta_min, beta_max=beta_max, data_ndim=data_ndim)

class ImageInterpXt(InterpXt):
    def __init__(self, args=None, N=1000, beta_min=0.1, beta_max=20, data_ndim=4):
        super().__init__(args=args, N=N, beta_min=beta_min, beta_max=beta_max, data_ndim=data_ndim)
        
    def _reshape_for_image(self, tensor):
        """Reshape coefficients to [BS, 1, 1, 1] for broadcasting with image data"""
        if tensor.dim() == 1:  # [BS]
            return tensor.view(-1, 1, 1, 1)
        elif tensor.dim() == 2:  # [BS, 1]
            return tensor.view(-1, 1, 1, 1)
        return tensor

    def get_alpha_t_beta_t(self, t, path_type=None):
        """Get alpha_t and beta_t reshaped for image broadcasting"""
        alpha_t, beta_t = super().get_alpha_t_beta_t(t, path_type=path_type)
        return self._reshape_for_image(alpha_t), self._reshape_for_image(beta_t)
    
    def get_d_alpha_beta_dt(self, t, alpha_t=None, beta_t=None):
        """Get derivatives reshaped for image broadcasting"""
        d_alpha_dt, d_beta_dt = super().get_d_alpha_beta_dt(t, alpha_t, beta_t)
        return self._reshape_for_image(d_alpha_dt), self._reshape_for_image(d_beta_dt)
    
    def get_coefficients_and_derivatives(self, t):
        """Get all coefficients and derivatives reshaped for image broadcasting"""
        alpha_t, beta_t, d_alpha_dt, d_beta_dt = super().get_coefficients_and_derivatives(t)
        return self._reshape_for_image(alpha_t), self._reshape_for_image(beta_t), self._reshape_for_image(d_alpha_dt), self._reshape_for_image(d_beta_dt)

