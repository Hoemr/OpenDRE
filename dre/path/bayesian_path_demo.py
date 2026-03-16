# -*- coding: utf-8 -*-
"""Demo and test code for Bayesian path (MVP, KMM): ConcatSquashExtendLinear, JointScoreModel, TestModel, visualization and training."""
import copy
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from dre.path import bayesian_path
from dre.losses.score_matching import TimeSampler

KMMBayesianTrajectory = bayesian_path.KMMBayesianTrajectory
def stopgrad(x):
    return x.detach()

class ConcatSquashExtendLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatSquashExtendLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper_gate(t.view(-1, 1))) + self._hyper_bias(t.view(-1, 1))


def unsqueeze(input, upscale_factor=2):
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels = in_channels // (upscale_factor**2)
    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor
    input_view = input.contiguous().view(batch_size, out_channels, upscale_factor, upscale_factor, in_height, in_width)
    output = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    return output.view(batch_size, out_channels, out_height, out_width)


def squeeze(input, downscale_factor=2):
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
    def __init__(self, hidden_dims, input_shape, path_model=None, strides=None, conv=False,
                  derivative_type="data_derivative", layer_type="concat", nonlinearity="softplus",
                  num_squeeze=0, args=None):
        super().__init__()
        self.num_squeeze = num_squeeze
        self.path_model = path_model
        self.eps = 1e-5
        self.derivative_type = derivative_type
        strides = [None] * (len(hidden_dims) + 1)
        base_layer = ConcatSquashExtendLinear
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
            raise ValueError("derivative_type error : %s" % derivative_type)
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
                raise ValueError("Unsupported stride: %s" % stride)
            layer = base_layer(hidden_shape[0], dim_out)
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
        for _ in range(self.num_squeeze):
            dx = squeeze(dx, 2)
        for l, layer in enumerate(self.layers):
            dx = layer(t, dx)
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        for _ in range(self.num_squeeze):
            dx = unsqueeze(dx, 2)
        if self.derivative_type == "joint_derivative":
            dx, dt = torch.split(dx, [y.shape[1], dx.shape[1] - y.shape[1]], dim=1)
            return dx, dt
        return dx


class TestModel(nn.Module):
    def __init__(self, beta_min=0.1, beta_max=20, eps=1e-5, path_type="VP", input_dim=1, joint=True, device="cpu", bridge=False):
        super().__init__()
        self.beta_0, self.beta_1 = beta_min, beta_max
        self.bridge, self.gamma_t = bridge, 1.
        self.eps, self.path_type = eps, path_type
        input_shape = (input_dim,)
        self.score_model = JointScoreModel(
            hidden_dims=[64, 64, 64], input_shape=input_shape,
            derivative_type="joint_derivative" if joint else "time_derivative",
        ).to(device)
        self.mvp = True
        if path_type == "mvp":
            self.bayesian_path = KMMBayesianTrajectory(
                n_components=10, constraint_type="spherical", eps=eps, grid_size=200
            ).to(device)
        self.step = 0
        weight_fn = lambda t: self.get_weights_denoise(t, mean=True)[0]
        self.t_mode = "AIS"
        self.t_sampler = TimeSampler(weights_fn=weight_fn, t_mode=self.t_mode, uniform_eps=eps, device=device)

    def forward(self, samples):
        x0, x1 = samples
        device = x0.device
        batch_size, dim = x0.size(0), x0.numel() // x0.size(0)
        t = self.t_sampler.sample_t(batch_size, device=device).to(device)
        x0, x1, t = x0.to(device), x1.to(device), t.to(device)
        x0_norm_sq = x0.flatten(1).pow(2).sum(1, keepdim=True)
        x1_norm_sq = x1.flatten(1).pow(2).sum(1, keepdim=True)
        x0x1_norm_sq = (x0 * x1).flatten(1).sum(1, keepdim=True)
        alpha_t, beta_t, d_alpha_dt, d_beta_dt = self.get_coefficients_and_derivatives(t)
        xt = self.marginal_sample(x0, x1, t, alpha_t=alpha_t, beta_t=beta_t)
        score_x, score_t = self.score_model(t, xt)
        if self.path_type == "mvp":
            variance = (2 * dim * d_alpha_dt**2 + x1_norm_sq * d_beta_dt**2) / (alpha_t**2 + 1e-5)
            elbo = self.bayesian_path.elbo(x0, x1, t, variance=variance, score_reg=score_t**2)
        if self.t_mode == "IS":
            lambda_t_time = torch.ones_like(t, device=device)
            lambda_t_data = stopgrad(2 * dim * d_alpha_dt**2 + x1_norm_sq * d_beta_dt**2)
        else:
            marginal_var = alpha_t ** 2
            marginal_var_dt = 2 * alpha_t * d_alpha_dt
            norm_term = d_beta_dt**2 * x1_norm_sq
            time_score_var = 0.5 * dim * (marginal_var_dt / marginal_var)**2 + norm_term / marginal_var
            lambda_t_time = 1. / time_score_var
            lambda_t_data = 1. / (marginal_var)
        target_cond_time_score = (- dim * d_alpha_dt + x0x1_norm_sq * d_beta_dt + x0_norm_sq * d_alpha_dt) / (alpha_t + 1e-5)
        time_loss = (score_t - stopgrad(target_cond_time_score))**2 * lambda_t_time
        diff = score_x - stopgrad(- x0 / (alpha_t + 1e-8))
        ssm_loss = torch.sum(diff ** 2, dim=tuple(range(1, diff.ndim)), keepdim=True) * lambda_t_data
        loss = time_loss + ssm_loss
        if self.mvp:
            loss += -elbo
            if self.step % 100 == 0:
                print("\n Loss: {:.2f}, time: {:.2f}, ssm: {:.2f}, elbo: {:.2f}".format(
                    loss.detach().mean().item(), time_loss.detach().mean().item(),
                    ssm_loss.detach().mean().item(), elbo.detach().mean().item()))
        self.step += 1
        return loss

    def get_weights_denoise(self, t, alpha_t=None, beta_t=None, d_alpha_dt=None, d_beta_dt=None, mean=False):
        if alpha_t is None:
            alpha_t, beta_t, d_alpha_dt, d_beta_dt = self.get_coefficients_and_derivatives(t, mean=mean)
        dim = 1.
        x1_norm_sq = 1.
        return alpha_t**2 / (2 * dim * d_alpha_dt**2 + x1_norm_sq * d_beta_dt**2), alpha_t**2

    def get_coefficients_and_derivatives(self, t, mean=False):
        if self.path_type == "mvp":
            coef = self.bayesian_path.sample_at_time(t, mode="posterior", return_derivs=True, mean=mean)
            return coef["alpha"], coef["beta"], coef["dot_alpha"], coef["dot_beta"]
        alpha_t, beta_t = self.get_alpha_t_beta_t(t, path_type=self.path_type)
        d_alpha_dt, d_beta_dt = self.get_d_alpha_beta_dt(t, alpha_t=alpha_t, beta_t=beta_t, path_type=self.path_type)
        return alpha_t, beta_t, d_alpha_dt, d_beta_dt

    def get_alpha_t_beta_t(self, t, path_type=None):
        ct = path_type or self.path_type
        if ct == "linear":
            return 1 - t, t
        if ct == "VP":
            log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
            alpha_t = torch.exp(log_mean_coeff)
            return alpha_t, torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        if ct == "follmer":
            return torch.sqrt(1 - t ** 2), t
        if ct == "cosine":
            s = torch.tensor(0.008, dtype=t.dtype, device=t.device)
            fn_t = torch.cos((t + s) / (1 + s) * math.pi / 2)
            fn_0 = torch.cos(s / (1 + s) * math.pi / 2)
            alpha_t_bar = fn_t / fn_0
            return torch.sqrt(alpha_t_bar), torch.sqrt(1 - alpha_t_bar)
        if ct == "trigonometric":
            x = math.pi * t * 0.5
            return torch.cos(x), torch.sin(x)
        if ct == "mvp":
            coef = self.bayesian_path.sample_at_time(t, mode="posterior", return_derivs=True)
            return coef["alpha"].view(-1, 1), coef["beta"].view(-1, 1)
        raise ValueError("Path type %s not implemented" % ct)

    def get_d_alpha_beta_dt(self, t, alpha_t=None, beta_t=None, path_type=None):
        ct = path_type or self.path_type
        if alpha_t is None:
            alpha_t, beta_t = self.get_alpha_t_beta_t(t, path_type=ct)
        if ct == "linear":
            return torch.full_like(t, -1.), torch.ones_like(t)
        if ct == "VP":
            term = 0.5 * ((self.beta_1 - self.beta_0) * t + self.beta_0)
            return -term * alpha_t, alpha_t ** 2 / (beta_t + 1e-6) * term
        if ct == "follmer":
            return -t / (torch.sqrt(1 - t**2) + 1e-6), torch.ones_like(t)
        if ct == "cosine":
            s = torch.tensor(0.008, dtype=t.dtype, device=t.device)
            const = math.pi * 0.5 / (1 + s)
            theta_t = (t + s) * const
            fn_0 = torch.cos(s * const)
            d_alpha_bar_dt = -torch.sin(theta_t) * const / fn_0
            return 0.5 * d_alpha_bar_dt / alpha_t, -0.5 * d_alpha_bar_dt / beta_t
        if ct == "trigonometric":
            return -math.pi / 2.0 * beta_t, math.pi / 2.0 * alpha_t
        if ct == "mvp":
            coef = self.bayesian_path.sample_at_time(t, mode="posterior", return_derivs=True)
            return coef["dot_alpha"], coef["dot_beta"]
        raise ValueError("Path type %s not implemented" % ct)

    def marginal_sample(self, x0, x1, t, alpha_t=None, beta_t=None):
        if alpha_t is None:
            alpha_t, beta_t = self.get_alpha_t_beta_t(t)
        return alpha_t * x0 + beta_t * x1


def get_alpha_t_beta_t(t, path_type="VP"):
    if path_type == "linear":
        return 1 - t, t
    if path_type == "VP":
        beta_0, beta_1 = 0.1, 20
        log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
        alpha_t = np.exp(log_mean_coeff)
        return alpha_t, np.sqrt(np.maximum(1. - np.exp(2. * log_mean_coeff), 1e-8))
    if path_type == "cosine":
        s = 0.008
        fn = lambda x: np.cos((x + s) / (1 + s) * np.pi / 2)
        alpha_t_bar = fn(t) / fn(0)
        return np.sqrt(alpha_t_bar), np.sqrt(np.maximum(1 - alpha_t_bar, 1e-8))
    if path_type == "follmer":
        return np.sqrt(np.maximum(1 - t ** 2, 1e-8)), t
    if path_type == "trigonometric":
        x = np.pi * t * 0.5
        return np.cos(x), np.sin(x)
    raise ValueError("Not implemented path type: %s" % path_type)


def get_d_alpha_beta_dt(t, alpha_t, beta_t, path_type="VP"):
    if path_type == "linear":
        return -np.ones_like(t), np.ones_like(t)
    if path_type == "VP":
        beta_0, beta_1 = 0.1, 20
        term = 0.5 * ((beta_1 - beta_0) * t + beta_0)
        return -term * alpha_t, (alpha_t ** 2 / beta_t) * term
    if path_type == "cosine":
        s, const = 0.008, np.pi * 0.5 / 1.008
        d_alpha_bar_dt = -np.sin((t + s) * const) * const / np.cos(s * const)
        return 0.5 * d_alpha_bar_dt / alpha_t, -0.5 * d_alpha_bar_dt / beta_t
    if path_type == "follmer":
        return -t / alpha_t, np.ones_like(t)
    if path_type == "trigonometric":
        return -np.pi / 2.0 * beta_t, np.pi / 2.0 * alpha_t
    raise ValueError("Not implemented path type: %s" % path_type)


def visualize_path_function(model, num_samples, prior_traj=None, post_traj=None, name="alpha", t_grid=None, sample_path="./"):
    if prior_traj is None:
        prior_traj = model.bayesian_path.sample_trajectories(mode="prior")
        post_traj = model.bayesian_path.sample_trajectories(mode="posterior")
        t_grid = model.bayesian_path.grid.cpu()
    alpha_prior = prior_traj[name].detach().cpu()
    alpha_post = post_traj[name].detach().cpu()
    ylabel = {"alpha": "$\\alpha_t$", "beta": "$\\beta_t$", "gamma": "$\\gamma_t$"}[name]
    plt.figure(figsize=(8, 6))
    methods = ['linear', 'VP', 'cosine', 'follmer', 'trigonometric']
    colors = {'linear': 'green', 'VP': 'black', 'cosine': 'orange', 'follmer': 'pink', 'trigonometric': 'purple'}
    labels = {'linear': 'Linear', 'VP': 'VP', 'cosine': 'Cosine', 'follmer': 'Follmer', 'trigonometric': 'Trigonometric'}
    t_grid_np = t_grid.cpu().numpy() if hasattr(t_grid, 'cpu') else t_grid
    if name != "gamma":
        for m in methods:
            ma, mb = get_alpha_t_beta_t(t_grid_np, path_type=m)
            if name == "alpha":
                plt.plot(t_grid_np, ma, color=colors[m], lw=2, linestyle='-', label=labels[m])
            elif name == "beta":
                plt.plot(t_grid_np, mb, color=colors[m], lw=2, linestyle='-', label=labels[m])
    plt.plot(t_grid_np, alpha_prior, 'b-', lw=2.5, label='KMM (prior, ours)')
    plt.plot(t_grid_np, alpha_post, 'r-', lw=2.5, label='KMM (posterior, ours)')
    plt.title('Trajectory Comparison (%s)' % ylabel, fontsize=16)
    plt.xlabel('Time $t$', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


def calculate_simplified_variance(t, alpha_t, beta_t, dot_alpha_t, dot_beta_t, gamma=1.0, epsilon=1e-4, bridge=False):
    if bridge:
        sigma_t_sq = t * (1 - t) * gamma**2 + (alpha_t**2 + beta_t**2) * epsilon
        dot_sigma_t_sq = (1 - 2 * t) * gamma**2 + (2 * alpha_t * dot_alpha_t + 2 * beta_t * dot_beta_t) * epsilon
        E_norm_sq = dot_alpha_t**2 + dot_beta_t**2 + 2 * dot_alpha_t * dot_beta_t
        return 0.5 * (dot_sigma_t_sq**2) / (sigma_t_sq**2) + E_norm_sq / sigma_t_sq
    return (2 * dot_alpha_t**2 + dot_beta_t**2) / (alpha_t**2 + epsilon)


def plot_variance_profiles(path_model, path_type="VP", bridge=False, gamma=1.0):
    fig, ax = plt.subplots(figsize=(8, 6))
    if path_type == "mvp":
        prior_traj = path_model.bayesian_path.sample_trajectories(mode="prior", return_derivs=True)
        post_traj = path_model.bayesian_path.sample_trajectories(mode="posterior", return_derivs=True)
        t_grid = path_model.bayesian_path.grid.cpu()
    else:
        eps = 1e-5
        t_grid = np.linspace(eps, 1 - eps, 500)
    methods = ['linear', 'VP', 'cosine', 'follmer', 'trigonometric']
    colors = {'linear': 'green', 'VP': 'black', 'cosine': 'orange', 'follmer': 'pink', 'trigonometric': 'purple'}
    labels = {'linear': 'Linear', 'VP': 'VP', 'cosine': 'Cosine', 'follmer': 'Follmer', 'trigonometric': 'Trigonometric'}
    t_grid_np = t_grid.cpu().numpy() if hasattr(t_grid, 'cpu') else t_grid
    for method in methods:
        alpha, beta = get_alpha_t_beta_t(t_grid_np, path_type=method)
        dot_alpha, dot_beta = get_d_alpha_beta_dt(t_grid_np, alpha, beta, path_type=method)
        variance = calculate_simplified_variance(t_grid_np, alpha, beta, dot_alpha, dot_beta, bridge=bridge, gamma=gamma)
        ax.plot(t_grid_np, variance, label=labels[method], color=colors[method], linewidth=2.5)
    if path_type == "mvp":
        alpha = post_traj["alpha"].detach().cpu().numpy()
        beta = post_traj["beta"].detach().cpu().numpy()
        dot_alpha = post_traj["dot_alpha"].detach().cpu().numpy()
        dot_beta = post_traj["dot_beta"].detach().cpu().numpy()
        variance = calculate_simplified_variance(t_grid_np, alpha, beta, dot_alpha, dot_beta, bridge=bridge, gamma=gamma)
        ax.plot(t_grid_np, variance, 'r-', lw=2.5, label="PAVE-DRE (ours)")
    ax.set_yscale('log')
    ax.set_title('Path-Dependent Variance of Different Interpolation Schedules', fontsize=18, pad=15)
    ax.set_xlabel('Time $t$', fontsize=14)
    ax.set_ylabel('Log Variance', fontsize=14)
    ax.legend(fontsize=12, loc='upper left')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(-0.05, 1.05)
    ax.axvspan(0.9, 1.02, color='red', alpha=0.1)
    ax.text(0.95, 1e-1, 'Potential Instability', rotation=90, verticalalignment='bottom', color='red', alpha=0.7, fontsize=14)
    plt.tight_layout()
    plt.savefig("variance_comparison_plot.pdf", dpi=300)
    plt.show()


def sample_from_mog(num_samples, gaussians):
    weights = np.array([g[2] for g in gaussians])
    weights /= weights.sum()
    samples = []
    for _ in range(num_samples):
        idx = np.random.choice(len(gaussians), p=weights)
        mean, std, _ = gaussians[idx]
        samples.append(np.random.normal(mean, std))
    return torch.tensor(samples).view(-1, 1)


def train(model, batch_size=500, device="cpu", epoch_num=1000):
    gaussians1 = [(-7, 1, 0.3), (-4, 1, 0.2), (2, 1, 0.2), (7, 1.5, 0.3)]
    gaussians2 = [(-2, 1.2, 0.5), (1, 1.5, 0.5)]
    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.001)
    for epoch in tqdm(range(epoch_num)):
        optimizer.zero_grad()
        x0 = sample_from_mog(batch_size, gaussians1)
        x1 = sample_from_mog(batch_size, gaussians2)
        if torch.cuda.is_available():
            x0, x1 = x0.to(device), x1.to(device)
        loss = model((x0, x1)).mean()
        if torch.isnan(loss).any():
            print("NaN detected in loss at epoch %s" % epoch)
            break
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TestModel(path_type="mvp", device=device)
    train(model, batch_size=1000, device=device, epoch_num=1000)
    num_samples = 5000
    prior_traj = model.bayesian_path.sample_trajectories(mode="prior")
    post_traj = model.bayesian_path.sample_trajectories(mode="posterior")
    t_grid = model.bayesian_path.grid.cpu().numpy()
    visualize_path_function(model, num_samples, prior_traj, post_traj, name="alpha", t_grid=t_grid)
    visualize_path_function(model, num_samples, prior_traj, post_traj, name="beta", t_grid=t_grid)
    plot_variance_profiles(model, path_type="mvp")
    t_points = torch.tensor([[0.1], [0.3], [0.5], [0.7], [0.7], [0.9]], device=device)
    samples = model.bayesian_path.sample_at_time(t_points, mode="posterior")
    print("============================================================")
    print("Sampled alpha: %s" % samples['alpha'].squeeze().tolist())
    print("Sampled beta: %s" % samples['beta'].squeeze().tolist())
