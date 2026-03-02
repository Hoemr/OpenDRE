# -*- coding: utf-8 -*-
"""
Activation functions, nonlinearity registry, and weight init methods.
Used by nn_models (neural network models) and optionally by pl_models.
"""
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init


class CosineAnnealingWarmRestartsDecay(optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing learning rate scheduler with warm restarts and decay.
    """
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, decay_factor=0.9, last_epoch=-1):
        self.T_0 = T_0  # Initial cycle length (in epochs)
        self.T_mult = T_mult  # Multiplicative factor to increase cycle length after each restart
        self.eta_min = eta_min  # Minimum learning rate
        self.decay_factor = decay_factor  # Factor to decay the maximum learning rate after each restart
        self.current_restart = 0  # Current number of restarts
        self.T_cur = 0  # Current step within the current cycle
        self.min_max_lr = 1.1 * self.eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Calculate the maximum learning rate for the current cycle (decaying over restarts)
        max_lr = self.base_lrs[0] * (self.decay_factor ** self.current_restart)
        max_lr = max(max_lr, self.min_max_lr) # Ensure max_lr is always greater than eta_min

        # Calculate the current learning rate (cosine annealing)
        return [
            self.eta_min + (max_lr - self.eta_min) * 
            (1 + math.cos(math.pi * self.T_cur / self.T_0)) / 2
            for _ in self.optimizer.param_groups
        ]

    def step(self):
        """Advance the scheduler by one step (called automatically in PyTorch Lightning)."""
        self.T_cur = self.T_cur + 1

        if self.T_cur >= self.T_0:
            # Restart cycle
            self.current_restart += 1
            self.T_cur = 0
            self.T_0 = int(self.T_0 * self.T_mult)  # Update cycle length

        super().step()


class Lambda(nn.Module):
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class TruncatedLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01, lower_bound=-5.0, upper_bound=1.0):
        super().__init__()
        self.upper_bound = nn.Parameter(torch.tensor(upper_bound))
        self.lower_bound = lower_bound
        self.alpha = 0.01
        self.beta = 0.1

    def forward(self, x):
        return torch.where(x < 0, self.alpha * x, torch.where(x <= self.upper_bound, x,
                                self.upper_bound + self.beta * (x - self.upper_bound)))


class DyT(nn.Module):
    def __init__(self, num_features, input_dim=2, alpha_init_value=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        if input_dim == 2:  # [BS, K]
            self.weight = nn.Parameter(torch.ones(1, num_features))
            self.bias = nn.Parameter(torch.zeros(1, num_features))
        elif input_dim == 4:  # [BS, K, 1, 1]
            self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        else:
            raise ValueError(f"Unsupported input dimension: {input_dim}. Supported dimensions are 2 and 4.")

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias


def init_dyt(module):
    if isinstance(module, DyT):
        init.constant_(module.alpha, 0.5)
        init.ones_(module.weight)
        init.zeros_(module.bias)


class TruncatedGELU(nn.Module):
    def __init__(self, lower_bound=-5.0, upper_bound=5.0):
        super().__init__()
        self.gelu = nn.GELU()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def forward(self, x):
        x = self.gelu(x)
        return torch.clamp(x, self.lower_bound, self.upper_bound)


class TruncatedELU(nn.Module):
    def __init__(self, alpha=1.0, upper_bound=10.0):
        super().__init__()
        self.alpha = alpha
        self.upper_bound = upper_bound

    def forward(self, x):
        elu_output = torch.where(x >= 0, x, self.alpha * (torch.exp(x) - 1))
        return torch.clamp(elu_output, max=self.upper_bound)


class ConvNormSilu(nn.Module):
    """Kernel fused function: Conv + GroupNorm + SiLU."""
    def __init__(self, conv_cls, num_groups=8, **conv_kwargs):
        super().__init__()
        num_groups = min(num_groups, conv_kwargs["out_channels"])
        self.layers = nn.ModuleList([
            conv_cls(**conv_kwargs),
            nn.GroupNorm(num_groups, conv_kwargs["out_channels"]),
            nn.SiLU(inplace=True),
        ])

    def forward(self, x, t_emb=None):
        for layer in self.layers:
            x = x.contiguous()
            x = layer(x)
        return x


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "dyt": lambda num_features, input_dim: DyT(num_features, input_dim=input_dim),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "gelu": nn.GELU(),
    "selu": nn.SELU(),
    "silu": nn.SiLU(),
    "leakyrelu": nn.LeakyReLU(),
    "prelu": nn.PReLU(),
    "swish": Swish(),
    "square": Lambda(lambda x: x**2),
    "identity": Lambda(lambda x: x),
    "truncleakyrelu": TruncatedLeakyReLU(),
    "truncgelu": TruncatedGELU(),
    "truncelu": TruncatedELU(),
}

INIT_METHODS = {
    "tanh": lambda w: init.xavier_normal_(w, gain=nn.init.calculate_gain('tanh')),
    "relu": lambda w: init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu'),
    "leakyrelu": lambda w: init.kaiming_normal_(w, mode='fan_in', nonlinearity='leaky_relu', a=0.01),
    "truncleakyrelu": lambda w: init.kaiming_normal_(w, mode='fan_in', nonlinearity='leaky_relu', a=0.01),
    "prelu": lambda w: init.kaiming_normal_(w, mode='fan_in', nonlinearity='leaky_relu', a=0.25),
    "elu": lambda w: init.kaiming_normal_(w, mode='fan_in', nonlinearity='linear'),
    "truncelu": lambda w: init.kaiming_normal_(w, mode='fan_in', nonlinearity='linear'),
    "gelu": lambda w: init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu'),
    "truncgelu": lambda w: init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu'),
    "selu": lambda w: init.lecun_normal_(w) * 1.5507,
    "silu": lambda w: init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu'),
    "softplus": lambda w: init.xavier_normal_(w, gain=1 / 0.89),
    "swish": lambda w: init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu'),
    "square": lambda w: init.xavier_normal_(w, gain=0.5),
    "identity": lambda w: init.xavier_normal_(w, gain=0.01),
    "dyt": init_dyt,
}

# for efficiency
ACTIVATION_TYPE_MAP = {type(act): name for name, act in NONLINEARITIES.items()}


def weights_init(m):
    """Initialize Linear/Conv weights and bias (used by HyperLinear, HyperConv2d)."""
    classname = m.__class__.__name__
    if classname.find("Linear") != -1 or classname.find("Conv") != -1:
        init.constant_(m.weight, 0)
        if hasattr(m, "bias") and m.bias is not None:
            init.normal_(m.bias, 0, 0.01)
