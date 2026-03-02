# -*- coding: utf-8 -*-
"""
Building blocks for neural network models: FusedConvBlock (image), ODEfunc (CNF).
"""
import numpy as np
import torch
import torch.nn as nn

from .activations import ConvNormSilu
from core.utils import divergence_approx, divergence_bf, sample_gaussian_like, sample_rademacher_like


class FusedConvBlock(nn.Module):
    """Three-in-one module with two Convolution layers, GroupNorm, and SiLU activation."""
    def __init__(self, in_ch, out_ch, embed_dim=None, stride=1, is_transpose=False, num_groups=8):
        super().__init__()
        conv_cls = nn.ConvTranspose2d if is_transpose else nn.Conv2d
        conv1_kwargs = {
            'in_channels': in_ch,
            'out_channels': in_ch,
            'kernel_size': 3,
            'stride': stride,
            'padding': 1,
            'bias': False
        }
        conv2_kwargs = {
            'in_channels': in_ch,
            'out_channels': out_ch,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1,
            'bias': False
        }
        if is_transpose:
            conv1_kwargs['output_padding'] = stride // 2

        self.layers = nn.ModuleList([
            ConvNormSilu(conv_cls, num_groups=num_groups, **conv1_kwargs),
            ConvNormSilu(conv_cls, num_groups=num_groups, **conv2_kwargs),
        ])
        self.time_proj = nn.Linear(embed_dim, out_ch) if embed_dim else None
        self.embed_dim = embed_dim
        self.out_ch = out_ch

    def forward(self, x, t_emb=None):
        for layer in self.layers:
            x = x.contiguous()
            x = layer(x)
        if t_emb is not None and self.time_proj:
            x = x + self.time_proj(t_emb)[..., None, None]
        return x


class ODEfunc(nn.Module):
    def __init__(self, diffeq, divergence_fn="approximate", residual=False, rademacher=False):
        super(ODEfunc, self).__init__()
        assert divergence_fn in ("brute_force", "approximate")
        self.diffeq = diffeq
        self.residual = residual
        self.rademacher = rademacher
        if divergence_fn == "brute_force":
            self.divergence_fn = divergence_bf
        elif divergence_fn == "approximate":
            self.divergence_fn = divergence_approx
        self.register_buffer("_num_evals", torch.tensor(0.))

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def num_evals(self):
        return self._num_evals.item()

    def forward(self, t, states):
        assert len(states) >= 2
        y = states[0]
        self._num_evals += 1
        t = t.clone().detach().to(y)
        batchsize = y.shape[0]
        if self._e is None:
            if self.rademacher:
                self._e = sample_rademacher_like(y)
            else:
                self._e = sample_gaussian_like(y)
        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            for s_ in states[2:]:
                s_.requires_grad_(True)
            dy = self.diffeq(t, y, *states[2:])
            if not self.training and dy.view(dy.shape[0], -1).shape[1] == 2:
                divergence = divergence_bf(dy, y).view(batchsize, 1)
            else:
                divergence = self.divergence_fn(dy, y, e=self._e).view(batchsize, 1)
        if self.residual:
            dy = dy - y
            divergence -= torch.ones_like(divergence) * torch.tensor(np.prod(y.shape[1:]), dtype=torch.float32).to(divergence)
        return tuple([dy, -divergence] + [torch.zeros_like(s_).requires_grad_(True) for s_ in states[2:]])
