import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv') != -1:
        nn.init.constant_(m.weight, 0)
        nn.init.normal_(m.bias, 0, 0.01)

class HyperLinear(nn.Module):
    def __init__(self, dim_in, dim_out, hypernet_dim=8, n_hidden=1, activation=nn.Tanh):
        super(HyperLinear, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.params_dim = self.dim_in * self.dim_out + self.dim_out

        layers = []
        dims = [1] + [hypernet_dim] * n_hidden + [self.params_dim]
        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i - 1], dims[i]))
            if i < len(dims) - 1:
                layers.append(activation())
        self._hypernet = nn.Sequential(*layers)
        self._hypernet.apply(weights_init)

    def forward(self, t, x):
        params = self._hypernet(t.view(1, 1)).view(-1)
        b = params[:self.dim_out].view(self.dim_out)
        w = params[self.dim_out:].view(self.dim_out, self.dim_in)
        return F.linear(x, w, b)


class IgnoreLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(IgnoreLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)

    def forward(self, t, x):
        return self._layer(x)


class ConcatLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(dim_in + 1, dim_out)

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ConcatLinear_v2(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)

    def forward(self, t, x):
        return self._layer(x) + self._hyper_bias(t.view(1, 1))


class SquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper(t.view(1, 1)))

class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper_gate(t.view(1, 1))) \
            + self._hyper_bias(t.view(1, 1))
            
class ConcatSquashExtendLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatSquashExtendLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper_gate(t.view(-1, 1))) \
            + self._hyper_bias(t.view(-1, 1))

class HyperConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(HyperConv2d, self).__init__()
        assert dim_in % groups == 0 and dim_out % groups == 0, "dim_in and dim_out must both be divisible by groups."
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.ksize = ksize
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.transpose = transpose

        self.params_dim = int(dim_in * dim_out * ksize * ksize / groups)
        if self.bias:
            self.params_dim += dim_out
        self._hypernet = nn.Linear(1, self.params_dim)
        self.conv_fn = F.conv_transpose2d if transpose else F.conv2d

        self._hypernet.apply(weights_init)

    def forward(self, t, x):
        params = self._hypernet(t.view(1, 1)).view(-1)
        weight_size = int(self.dim_in * self.dim_out * self.ksize * self.ksize / self.groups)
        if self.transpose:
            weight = params[:weight_size].view(self.dim_in, self.dim_out // self.groups, self.ksize, self.ksize)
        else:
            weight = params[:weight_size].view(self.dim_out, self.dim_in // self.groups, self.ksize, self.ksize)
        bias = params[:self.dim_out].view(self.dim_out) if self.bias else None
        return self.conv_fn(
            x, weight=weight, bias=bias, stride=self.stride, padding=self.padding, groups=self.groups,
            dilation=self.dilation
        )


class IgnoreConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(IgnoreConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        return self._layer(x)


class SquashConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(SquashConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
        self._hyper = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper(t.view(1, 1))).view(1, -1, 1, 1)


class ConcatConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ConcatConv2d_v2(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)

    def forward(self, t, x):
        return self._layer(x) + self._hyper_bias(t.view(1, 1)).view(1, -1, 1, 1)


class ConcatSquashConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatSquashConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
        self._hyper_gate = nn.Linear(1, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper_gate(t.view(1, 1))).view(1, -1, 1, 1) \
            + self._hyper_bias(t.view(1, 1)).view(1, -1, 1, 1)


class ConcatCoordConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatCoordConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 3, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        b, c, h, w = x.shape
        hh = torch.arange(h).to(x).view(1, 1, h, 1).expand(b, 1, h, w)
        ww = torch.arange(w).to(x).view(1, 1, 1, w).expand(b, 1, h, w)
        tt = t.to(x).view(1, 1, 1, 1).expand(b, 1, h, w)
        x_aug = torch.cat([x, tt, hh, ww], 1)
        return self._layer(x_aug)


class GatedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(GatedLinear, self).__init__()
        self.layer_f = nn.Linear(in_features, out_features)
        self.layer_g = nn.Linear(in_features, out_features)

    def forward(self, x):
        f = self.layer_f(x)
        g = torch.sigmoid(self.layer_g(x))
        return f * g


class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(GatedConv, self).__init__()
        self.layer_f = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=1, groups=groups
        )
        self.layer_g = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=1, groups=groups
        )

    def forward(self, x):
        f = self.layer_f(x)
        g = torch.sigmoid(self.layer_g(x))
        return f * g


class GatedConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1):
        super(GatedConvTranspose, self).__init__()
        self.layer_f = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding,
            groups=groups
        )
        self.layer_g = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding,
            groups=groups
        )

    def forward(self, x):
        f = self.layer_f(x)
        g = torch.sigmoid(self.layer_g(x))
        return f * g


class BlendLinear(nn.Module):
    def __init__(self, dim_in, dim_out, layer_type=nn.Linear, **unused_kwargs):
        super(BlendLinear, self).__init__()
        self._layer0 = layer_type(dim_in, dim_out)
        self._layer1 = layer_type(dim_in, dim_out)

    def forward(self, t, x):
        y0 = self._layer0(x)
        y1 = self._layer1(x)
        return y0 + (y1 - y0) * t


class BlendConv2d(nn.Module):
    def __init__(
        self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False,
        **unused_kwargs
    ):
        super(BlendConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer0 = module(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
        self._layer1 = module(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        y0 = self._layer0(x)
        y1 = self._layer1(x)
        return y0 + (y1 - y0) * t
    
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
            # Pre-norm：Norm → Activation → Linear → Dropout
            if self.is_shortcut:
                self.forward_func = lambda x: self.dropout(self.linear(self.activation(self.norm_layer(x)))) + self.shortcut(x)
            else:
                self.forward_func = lambda x: self.dropout(self.linear(self.activation(self.norm_layer(x)))) + x
        elif norm and not pre_norm:
            # Post-norm：Linear → Norm → Activation → Dropout
            if self.is_shortcut:
                self.forward_func = lambda x: self.dropout(self.activation(self.norm_layer(self.linear(x)))) + self.shortcut(x)
            else:
                self.forward_func = lambda x: self.dropout(self.activation(self.norm_layer(self.linear(x)))) + x
        else:
            # without norm：Linear → Activation → Dropout
            if self.is_shortcut:
                self.forward_func = lambda x: self.dropout(self.activation(self.linear(x))) + self.shortcut(x)
            else:
                self.forward_func = lambda x: self.dropout(self.activation(self.linear(x))) + x
        
    def forward(self, x):
        return self.forward_func(x)

class ResidualBlockWithTime(nn.Module):
    """ResidualBlock with time injection"""
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
        # Pre-norm：Norm → Activation → Linear → TimeInjection → Dropout
        residual = self.shortcut(x) if self.is_shortcut else x
        x = self.linear(self.activation(self.norm_layer(x)))
        x = x * torch.sigmoid(self._hyper_gate(time_emb)) + self._hyper_bias(time_emb)
        return self.dropout(x) + residual
    
    def _forward_post_norm(self, x, time_emb):
        # Post-norm：Linear → TimeInjection → Norm → Activation → Dropout
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
    
class FourierSinusoidalTimeEncoder(nn.Module):
    def __init__(self, output_dim, nfreq=256):
        super().__init__()
        # mid_dim = (nfreq + output_dim) // 4
        mid_dim = output_dim
        self.mlp = nn.Sequential(nn.Linear(nfreq, mid_dim), 
                                 nn.SiLU(), 
                                 nn.Linear(mid_dim, output_dim))
        self.nfreq = nfreq

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32)
            / half_dim
        ).to(device=t.device)
        args = t.float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t = t * 1000    # [BS, 1]
        t_freq = self.timestep_embedding(t, self.nfreq)  # [BS, nfreq]
        return self.mlp(t_freq)    # [BS, dim]
    
class SinusoidalTimeEncoder(nn.Module):
    def __init__(self, output_dim, nfreq=256):
        super().__init__()
        self.nfreq = nfreq
        self.mlp = nn.Sequential(
                nn.Linear(2*nfreq, output_dim),
                nn.SiLU(),
                nn.Linear(output_dim, output_dim)
            )

    def forward(self, x):
        emb = math.log(10000) / (self.nfreq - 1)
        emb = torch.exp(torch.arange(self.nfreq, device=x.device) * -emb)
        emb = x * emb[None, :]  
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return self.mlp(emb)

class FourierTimeEncoder(nn.Module):
    def __init__(self, output_dim, nfreq=256):
        super().__init__()
        if nfreq is None:
            nfreq = output_dim // 2  
        self.register_buffer('freq_bands', 2**torch.linspace(0, 8, nfreq))
        self.mlp = nn.Sequential(
            nn.Linear(2*nfreq, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, t):
        projections = t * self.freq_bands * 2 * math.pi  # [B, F]
        emb = torch.cat([torch.sin(projections), torch.cos(projections)], dim=-1) # [B,2F]
        return self.mlp(emb)  # [B, output_dim]

class GaussianFourierTimeEncoder(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, output_dim, nfreq=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(nfreq) * scale, requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(2*nfreq, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        x = x.squeeze(-1)
        x_proj = x[:, None] * self.W[None, :] * 2 * math.pi
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.mlp(emb)  # [B, output_dim]

class AdaptiveLastLayer(nn.Module):
    def __init__(self, in_dim, out_dim, embed_dim):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x, time_emb):
        return self.dense(x) * self.gate(time_emb)  # time-dependent gate control

class LinearLastLayer(nn.Module):
    def __init__(self, in_dim, out_dim, embed_dim=None):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)
        
    def forward(self, x, time_emb=None):
        return self.dense(x)


# =====================================================
# ResNet and WideResNet for Image-based OOD Detection
# =====================================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, out_dim=1, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = F.relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, out_dim)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def ResNet18(out_dim=1):
    """ResNet-18 model"""
    return ResNet(BasicBlock, [2, 2, 2, 2], out_dim=out_dim)


def ResNet34(out_dim=1):
    """ResNet-34 model"""
    return ResNet(BasicBlock, [3, 4, 6, 3], out_dim=out_dim)


def ResNet50(out_dim=1):
    """ResNet-50 model"""
    return ResNet(Bottleneck, [3, 4, 6, 3], out_dim=out_dim)


# =====================================================
# Wide ResNet for Image-based OOD Detection
# =====================================================

class WideResNetBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.3):
        super(WideResNetBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

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
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.3):
        super(WideResNetNetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_classes=10, dropRate=0.3):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = WideResNetBasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 3 network blocks
        self.block1 = WideResNetNetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = WideResNetNetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = WideResNetNetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # 4th batch norm
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        # fc
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
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


def WideResNet2810(out_dim=10, dropRate=0.3):
    """WideResNet-28-10 with configurable output dimension"""
    return WideResNet(depth=28, widen_factor=10, num_classes=out_dim, dropRate=dropRate)
