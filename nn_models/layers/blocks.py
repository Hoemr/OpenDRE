# -*- coding: utf-8 -*-
"""
Building blocks for neural network models: FusedConvBlock, ODEfunc, RMSNorm,
ResidualBlock, ResidualBlockWithTime, ResNet, WideResNet.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import ConvNormSilu
from dre_utils.utils import divergence_approx, divergence_bf, sample_gaussian_like, sample_rademacher_like


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


# ---------------------------------------------------------------------------
# Norm and residual blocks (tabular / score models)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return self.weight * x / (norm + self.eps)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, activation, dropout=0.5, norm=True, use_spectral_norm=False, pre_norm=True):
        super(ResidualBlock, self).__init__()
        if use_spectral_norm:
            self.linear = nn.utils.spectral_norm(nn.Linear(in_dim, out_dim))
        else:
            self.linear = nn.Linear(in_dim, out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        if norm:
            self.norm_layer = RMSNorm(in_dim if pre_norm else out_dim)
        self.is_shortcut = in_dim != out_dim
        self.shortcut = nn.Linear(in_dim, out_dim) if self.is_shortcut else None
        if norm and pre_norm:
            if self.is_shortcut:
                self.forward_func = lambda x: self.dropout(self.linear(self.activation(self.norm_layer(x)))) + self.shortcut(x)
            else:
                self.forward_func = lambda x: self.dropout(self.linear(self.activation(self.norm_layer(x)))) + x
        elif norm and not pre_norm:
            if self.is_shortcut:
                self.forward_func = lambda x: self.dropout(self.activation(self.norm_layer(self.linear(x)))) + self.shortcut(x)
            else:
                self.forward_func = lambda x: self.dropout(self.activation(self.norm_layer(self.linear(x)))) + x
        else:
            if self.is_shortcut:
                self.forward_func = lambda x: self.dropout(self.activation(self.linear(x))) + self.shortcut(x)
            else:
                self.forward_func = lambda x: self.dropout(self.activation(self.linear(x))) + x

    def forward(self, x):
        return self.forward_func(x)


class ResidualBlockWithTime(nn.Module):
    """ResidualBlock with time injection."""
    def __init__(self, in_dim, out_dim, embed_dim, activation, dropout=0.1, norm=True, use_spectral_norm=False, pre_norm=True):
        super().__init__()
        if use_spectral_norm:
            self.linear = nn.utils.spectral_norm(nn.Linear(in_dim, out_dim))
        else:
            self.linear = nn.Linear(in_dim, out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        if norm:
            self.norm_layer = RMSNorm(in_dim if pre_norm else out_dim)
        self.is_shortcut = in_dim != out_dim
        self.shortcut = nn.Linear(in_dim, out_dim) if self.is_shortcut else None
        self._hyper_bias = nn.Linear(embed_dim, out_dim, bias=False)
        self._hyper_gate = nn.Linear(embed_dim, out_dim)
        if norm and pre_norm:
            self.forward_func = self._forward_pre_norm
        elif norm and not pre_norm:
            self.forward_func = self._forward_post_norm
        else:
            self.forward_func = self._forward_no_norm

    def _forward_pre_norm(self, x, time_emb):
        residual = self.shortcut(x) if self.is_shortcut else x
        x = self.linear(self.activation(self.norm_layer(x)))
        x = x * torch.sigmoid(self._hyper_gate(time_emb)) + self._hyper_bias(time_emb)
        return self.dropout(x) + residual

    def _forward_post_norm(self, x, time_emb):
        residual = self.shortcut(x) if self.is_shortcut else x
        x = self.linear(x)
        x = x * torch.sigmoid(self._hyper_gate(time_emb)) + self._hyper_bias(time_emb)
        x = self.activation(self.norm_layer(x))
        return self.dropout(x) + residual

    def _forward_no_norm(self, x, time_emb):
        residual = self.shortcut(x) if self.is_shortcut else x
        x = self.linear(x)
        x = x * torch.sigmoid(self._hyper_gate(time_emb)) + self._hyper_bias(time_emb)
        x = self.activation(x)
        return self.dropout(x) + residual

    def forward(self, x, time_emb):
        return self.forward_func(x, time_emb)


# ---------------------------------------------------------------------------
# ResNet and WideResNet (image backbones)
# ---------------------------------------------------------------------------

def _make_norm2d(norm_type, num_channels, num_groups=8):
    """Return a 2D norm layer: BatchNorm2d or GroupNorm."""
    if norm_type == "groupnorm":
        return nn.GroupNorm(min(num_groups, num_channels), num_channels)
    return nn.BatchNorm2d(num_channels)


def _make_activation(nonlinearity):
    """Return inplace activation for ResNet blocks: ReLU or LeakyReLU."""
    if (nonlinearity or "").lower() == "leakyrelu":
        return nn.LeakyReLU(0.01, inplace=True)
    return nn.ReLU(inplace=True)


class BasicBlock(nn.Module):
    """
    BasicBlock with pre-activation style (matching OpenOOD).
    
    Pre-activation: Norm -> Act -> Conv, shortcut has no activation.
    Supports BatchNorm or GroupNorm, ReLU or LeakyReLU via norm_layer and activation.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm_layer=None, activation=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = lambda c: nn.BatchNorm2d(c)
        if activation is None:
            activation = nn.ReLU(inplace=True)
        self.bn1 = norm_layer(in_channels)
        self.relu1 = activation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)
        self.relu2 = activation
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = downsample
        self.equalInOut = (in_channels == out_channels and stride == 1)

    def forward(self, x):
        if not self.equalInOut:
            # When dimensions change, apply BN-ReLU to input for shortcut path
            out = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(out)))
            out = self.conv2(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            else:
                identity = x
        else:
            # When dimensions match, shortcut is direct
            out = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(out)))
            out = self.conv2(out)
            identity = x
        out += identity
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm_layer=None, activation=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = lambda c: nn.BatchNorm2d(c)
        if activation is None:
            activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.downsample = downsample
        self.activation = activation

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.activation(out)
        return out


class ResNet(nn.Module):
    """
    ResNet for small images (32x32) matching OpenOOD style.
    
    Key differences from standard ResNet:
    1. No initial MaxPool (preserves spatial info for small images)
    2. Uses pre-activation BasicBlock
    3. Final norm-act before pooling
    norm_type: "batchnorm" or "groupnorm". nonlinearity: "relu" or "leakyrelu".
    """
    def __init__(self, block, layers, out_dim=1, num_classes=1000, norm_type="batchnorm", nonlinearity="relu", num_groups=8):
        super(ResNet, self).__init__()
        self.in_channels = 64
        norm_layer = lambda c: _make_norm2d(norm_type, c, num_groups)
        self._activation = _make_activation(nonlinearity)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = self._activation
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer, self._activation)
        self.layer2 = self._make_layer(block, 128, layers[1], norm_layer, self._activation, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], norm_layer, self._activation, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], norm_layer, self._activation, stride=2)
        self.bn_final = norm_layer(512 * block.expansion)
        self.relu_final = self._activation
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, out_dim)
        _init_resnet_weights(self, nonlinearity)

    def _make_layer(self, block, out_channels, blocks, norm_layer, activation, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels * block.expansion),
            )
        layer_list = [block(self.in_channels, out_channels, stride, downsample, norm_layer=norm_layer, activation=activation)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layer_list.append(block(self.in_channels, out_channels, norm_layer=norm_layer, activation=activation))
        return nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # No maxpool for 32x32 images
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Final BN-ReLU before pooling (pre-activation style)
        x = self.bn_final(x)
        x = self.relu_final(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def _init_resnet_weights(module, nonlinearity="relu"):
    """Kaiming init for Conv2d; init for BatchNorm2d/GroupNorm."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu" if nonlinearity == "relu" else "leaky_relu", a=0.01)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def ResNet18(out_dim=1, norm_type="batchnorm", nonlinearity="relu", num_groups=8):
    return ResNet(BasicBlock, [2, 2, 2, 2], out_dim=out_dim, norm_type=norm_type, nonlinearity=nonlinearity, num_groups=num_groups)


def ResNet34(out_dim=1, norm_type="batchnorm", nonlinearity="relu", num_groups=8):
    return ResNet(BasicBlock, [3, 4, 6, 3], out_dim=out_dim, norm_type=norm_type, nonlinearity=nonlinearity, num_groups=num_groups)


def ResNet50(out_dim=1, norm_type="batchnorm", nonlinearity="relu", num_groups=8):
    return ResNet(Bottleneck, [3, 4, 6, 3], out_dim=out_dim, norm_type=norm_type, nonlinearity=nonlinearity, num_groups=num_groups)


class WideResNetBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.3, norm_layer=None, activation=None):
        super(WideResNetBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = lambda c: nn.BatchNorm2d(c)
        if activation is None:
            activation = nn.ReLU(inplace=True)
        self.bn1 = norm_layer(in_planes)
        self.relu1 = activation
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(out_planes)
        self.relu2 = activation
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class WideResNetNetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.3, norm_layer=None, activation=None):
        super(WideResNetNetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, norm_layer, activation)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, norm_layer, activation):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, norm_layer=norm_layer, activation=activation))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_classes=10, dropRate=0.3, norm_type="batchnorm", nonlinearity="relu", num_groups=8):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = WideResNetBasicBlock
        norm_layer = lambda c: _make_norm2d(norm_type, c, num_groups)
        activation = _make_activation(nonlinearity)
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = WideResNetNetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, norm_layer, activation)
        self.block2 = WideResNetNetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, norm_layer, activation)
        self.block3 = WideResNetNetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, norm_layer, activation)
        self.bn1 = norm_layer(nChannels[3])
        self.relu = activation
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        _init_resnet_weights(self, nonlinearity)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


def WideResNet2810(out_dim=10, dropRate=0.3, norm_type="batchnorm", nonlinearity="relu", num_groups=8):
    return WideResNet(depth=28, widen_factor=10, num_classes=out_dim, dropRate=dropRate, norm_type=norm_type, nonlinearity=nonlinearity, num_groups=num_groups)
