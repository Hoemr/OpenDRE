# -*- coding: utf-8 -*-
import math
import itertools

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import solve
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist
from sklearn.model_selection import GridSearchCV
from scipy.spatial.distance import cdist


from . import layers as Layers
from . import polynomials as Pk
from .pl_data import DATA_SHAPES
from utils.utils import _flip
from utils.utils import unsqueeze, squeeze
from utils.utils import divergence_approx, divergence_bf
from utils.utils import sample_gaussian_like, sample_rademacher_like
import numpy as np


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
        # self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        # self.lower_bound = nn.Parameter(torch.tensor(lower_bound))
        self.upper_bound = nn.Parameter(torch.tensor(upper_bound))
        self.lower_bound = lower_bound
        # self.upper_bound = upper_bound
        
        self.alpha = 0.01
        self.beta = 0.1

    def forward(self, x):
        return torch.where(x < 0, self.alpha * x, torch.where(x <=self.upper_bound, x,
                                self.upper_bound + self.beta * (x - self.upper_bound)))
        # return torch.clamp(self.leaky_relu(x), self.lower_bound, self.upper_bound)
        
class DyT(nn.Module):
    def __init__(self, num_features, input_dim=2, alpha_init_value=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)  # Learnable alpha parameter
        if input_dim == 2:  # [BS, K]
            self.weight = nn.Parameter(torch.ones(1, num_features))
            self.bias = nn.Parameter(torch.zeros(1, num_features))
        elif input_dim == 4:  # [BS, K, 1, 1]
            self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        else:
            raise ValueError(f"Unsupported input dimension: {input_dim}. Supported dimensions are 2 and 4.")
        
    def forward(self, x):
        x = torch.tanh(self.alpha * x)  # Apply tanh with learnable alpha
        return x * self.weight + self.bias  # Scale and shift with learnable parameters

def init_dyt(module):
    if isinstance(module, DyT):
        init.constant_(module.alpha, 0.5)  # 初始化alpha为0.5
        init.ones_(module.weight)  # 初始化weight为1
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
    """Kernel fused function"""
    def __init__(self, conv_cls, num_groups=8, **conv_kwargs):
        super().__init__()
        num_groups=min(num_groups, conv_kwargs["out_channels"])
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
    "gelu": lambda w: init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu'),  # GELU近似ReLU初始化
    "truncgelu": lambda w: init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu'),
    "selu": lambda w: init.lecun_normal_(w) * 1.5507,  # 配合自归一化网络的scale系数
    "silu": lambda w: init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu'),
    "softplus": lambda w: init.xavier_normal_(w, gain=1 / 0.89),  # Softplus增益近似0.89
    "swish": lambda w: init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu'),  # 近似ReLU初始化
    "square": lambda w: init.xavier_normal_(w, gain=0.5),  # 平方函数需降低增益防止梯度爆炸
    "identity": lambda w: init.xavier_normal_(w, gain=0.01),
    "dyt": init_dyt,
}

# for efficiency
ACTIVATION_TYPE_MAP = {type(act): name for name, act in NONLINEARITIES.items()}


# ==================== Unified Model Architecture ====================

class UnifiedModel(nn.Module):
    """
    A unified model that can handle both image and non-image data with different architectures and tasks.
    
    Args:
        model_type: 'joint_score', 'decomp_joint_score', or 'density_ratio'
        data_type: 'tabular' or 'image'
        backbone: For image data - 'unet', 'resnet18', 'resnet34', 'resnet50', 'wideresnet28_10'
                  For tabular data - 'mlp'
        input_dim: Input dimension for tabular data
        in_channels: Number of input channels for image data
        hidden_dims: Hidden dimensions for tabular data
        hidden_channels: Hidden channel dimensions for image data
        K: Number of polynomial components for decomposition (only for decomp_joint_score)
        embed_dim: Time embedding dimension
        nonlinearity: Activation function
        polynomials: Type of polynomials for decomposition
        noise_type: Noise type for polynomials
        args: Additional arguments
        norm: Whether to use normalization
        use_spectral_norm: Whether to use spectral normalization
        dynamic: Whether to use dynamic layers
        dropout: Dropout rate for WideResNet
    """
    
    def __init__(self, model_type='joint_score', data_type='tabular', backbone='mlp',
                 input_dim=None, in_channels=None, hidden_dims=None, hidden_channels=None,
                 K=300, embed_dim=256, nonlinearity="leakyrelu", polynomials="gaussrbf", 
                 noise_type="zero", args=None, norm=False, use_spectral_norm=False, 
                 dynamic=False, dropout=0.3):
        super().__init__()
        
        self.model_type = model_type
        self.data_type = data_type
        self.backbone = backbone
        self.args = args
        self.joint = args.joint if args else False
        self.K = K
        self.embed_dim = embed_dim
        self.nonlinearity = nonlinearity
        
        # Initialize polynomial decomposition if needed
        if model_type == 'decomp_joint_score':
            self.Pk = Pk.PerturbedFrame(K, noise_type=noise_type, polynomials=polynomials, 
                                       shape_type=getattr(args, 'shape_type', 'gauss'))
            self.set_fk()
        
        # Build the appropriate architecture based on data type and backbone
        if data_type == 'tabular':
            self._build_tabular_architecture(input_dim, hidden_dims, embed_dim, nonlinearity, 
                                           args, norm, use_spectral_norm, dynamic)
        elif data_type == 'image':
            self._build_image_architecture(in_channels, hidden_channels, embed_dim, 
                                         nonlinearity, dropout, args)
        
        # Determine forward method based on model configuration to avoid repeated if statements
        if model_type in ['joint_score', 'decomp_joint_score']:
            if data_type == 'tabular':
                self._forward_time_dependent_impl = self._forward_tabular_time_dependent
                self._compute_s_t_impl = self._compute_s_t_tabular_decomp if model_type == 'decomp_joint_score' else self._compute_s_t_tabular_joint
            elif data_type == 'image':
                if backbone in ['resnet18', 'resnet34', 'resnet50', 'wideresnet28_10']:
                    self._forward_time_dependent_impl = self._forward_resnet_time_dependent
                    self._compute_s_t_impl = self._compute_s_t_resnet_decomp if model_type == 'decomp_joint_score' else self._compute_s_t_resnet_joint
                else:  # UNet
                    self._forward_time_dependent_impl = self._forward_unet_time_dependent
                    self._compute_s_t_impl = self._compute_s_t_unet_decomp if model_type == 'decomp_joint_score' else self._compute_s_t_unet_joint
            self.forward_impl = self._forward_time_dependent_impl
        elif model_type == 'density_ratio':
            if data_type == 'tabular':
                self._forward_density_ratio_impl = self._forward_density_ratio_tabular
            elif data_type == 'image':
                if backbone in ['resnet18', 'resnet34', 'resnet50', 'wideresnet28_10']:
                    self._forward_density_ratio_impl = self._forward_density_ratio_image_resnet
                else:
                    self._forward_density_ratio_impl = self._forward_density_ratio_image_unet
            self.forward_impl = self._forward_density_ratio_impl
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Determine compute_fk implementation based on data type and backbone
        if data_type == 'tabular':
            self.compute_fk_impl = self._compute_fk_tabular
        elif data_type == 'image':
            if backbone == 'unet':
                self.compute_fk_impl = self._compute_fk_unet
            else:  # ResNet/WideResNet
                self.compute_fk_impl = self._compute_fk_resnet
        
        # Pre-determine additional flags to avoid repeated if statements in forward pass
        self._is_decomp_joint_score = (model_type == 'decomp_joint_score')
        self._is_joint_score = (model_type == 'joint_score')
        self._is_time_dependent = (model_type in ['joint_score', 'decomp_joint_score'])
        
        # Pre-compute image dimensions for ResNet architecture (used in forward)
        if data_type == 'image' and backbone not in ['unet']:
            data_shape = getattr(args, 'data', None) if args else None
            if data_shape in DATA_SHAPES:
                self._img_channels, self._img_height, self._img_width = DATA_SHAPES[data_shape]
            else:
                self._img_channels, self._img_height, self._img_width = in_channels, 32, 32
        
        self._initialize_weights()
        self.nfe = 0  # Number of function evaluations
        
    def _build_tabular_architecture(self, input_dim, hidden_dims, embed_dim, nonlinearity, 
                                  args, norm, use_spectral_norm, dynamic):
        """Build architecture for tabular data."""
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        dropout = getattr(args, 'dropout', 0.0) if args else 0.0
        pre_norm = getattr(args, 'pre_norm', False) if args else False
        timeemb_outdim = input_dim
        
        def build_dynamic_layers(input_dim=None, output_dim=None, num_layers=None, 
                                 dim_sequence=None, activation="leakyrelu", 
                                 max_expansion=4.0, min_layers=1):
            if dim_sequence is None:
                if dynamic:
                    if num_layers is None:
                        if input_dim == output_dim:
                            num_layers = min_layers
                        else:
                            ratio = output_dim / input_dim
                            num_layers = max(min_layers, int(np.ceil(np.log(ratio) / np.log(max_expansion))))

                    dim_sequence = list(np.geomspace(input_dim, output_dim, num=num_layers + 1, dtype=int))
                else:
                    dim_sequence = [input_dim] + [input_dim]*num_layers + [output_dim]
            
            layers = []
            for i in range(len(dim_sequence) - 1):
                in_dim, out_dim = dim_sequence[i], dim_sequence[i + 1]
                if self.model_type == 'joint_score':
                    # For joint score models, use time-aware blocks
                    layers.append(Layers.ResidualBlockWithTime(
                        in_dim, out_dim, 
                        embed_dim=timeemb_outdim,
                        activation=self._get_activation(activation), 
                        dropout=dropout,
                        norm=norm,
                        use_spectral_norm=use_spectral_norm,
                        pre_norm=pre_norm,))
                else:
                    # For other models, use regular blocks
                    layers.append(Layers.ResidualBlock(
                        in_dim, out_dim, 
                        activation=self._get_activation(activation), 
                        dropout=dropout,
                        norm=norm,
                        use_spectral_norm=use_spectral_norm,
                        pre_norm=pre_norm,))
            return layers

        # Shared network 
        dim_sequence = [input_dim] + hidden_dims
        self.shared_layers = nn.Sequential(
            *build_dynamic_layers(dim_sequence=dim_sequence, 
                                  activation=nonlinearity)
        )

        # Time heads
        if self.model_type == 'decomp_joint_score':
            # For decomposition models, output K components
            self.last_layer_time = Layers.LinearLastLayer(hidden_dims[-1], self.K)
            self.time_head = nn.Sequential(
                *build_dynamic_layers(input_dim=hidden_dims[-1], 
                                      output_dim=hidden_dims[-1], 
                                      num_layers=getattr(args, 'num_time_head', 1)-1, 
                                      activation=nonlinearity),   
                self.last_layer_time,
            )
        else:
            # For other models, output single value (or for density ratio)
            output_dim = 1
            if self.model_type == 'density_ratio':
                # For density ratio, we might want to output a single value
                output_dim = 1
            else:
                # For joint score, output single time score
                output_dim = 1
                
            self.last_layer_time = Layers.LinearLastLayer(hidden_dims[-1], output_dim)
            self.time_head = nn.Sequential(
                *build_dynamic_layers(input_dim=hidden_dims[-1], 
                                      output_dim=hidden_dims[-1], 
                                      num_layers=getattr(args, 'num_time_head', 1)-1, 
                                      activation=nonlinearity),   
                self.last_layer_time,
            )

        # Data head: Fourier time encoding (only for joint models)
        if self.joint and self.model_type in ['joint_score', 'decomp_joint_score']:
            if self.model_type == 'joint_score':
                last_layer_data = Layers.LinearLastLayer(hidden_dims[-1], input_dim)
                self.data_head = nn.Sequential(
                    *build_dynamic_layers(input_dim=hidden_dims[-1], 
                                        output_dim=hidden_dims[-1], 
                                        num_layers=getattr(args, 'num_time_head', 1)-1, 
                                        activation=nonlinearity),   
                    last_layer_data,
                )
            elif self.model_type == 'decomp_joint_score':
                last_layer_data = Layers.LinearLastLayer(hidden_dims[-1], input_dim)
                self.data_head = nn.Sequential(
                    *[Layers.ResidualBlockWithTime(
                        in_dim=hidden_dims[-1], out_dim=hidden_dims[-1],
                        embed_dim=timeemb_outdim,  # time_embed shape = input data shape
                        activation=self._get_activation(nonlinearity),
                        dropout=dropout,
                        norm=norm,
                        use_spectral_norm=use_spectral_norm,
                        pre_norm=pre_norm,
                      ) for _ in range(getattr(args, 'num_data_head', 1)-1)],
                    last_layer_data
                )

        self.time_emb = Layers.FourierSinusoidalTimeEncoder(output_dim=timeemb_outdim, nfreq=embed_dim)
        # self.delta_emb = Layers.FourierSinusoidalTimeEncoder(output_dim=timeemb_outdim, nfreq=embed_dim)
        if getattr(args, 'SAI', False) if args else False:
            self.time_fusion = nn.Sequential(
                nn.Linear(timeemb_outdim * 2, timeemb_outdim),  
                nn.SiLU(),
                nn.Linear(timeemb_outdim, timeemb_outdim)    
            )
    
    def _build_image_architecture(self, in_channels, hidden_channels, embed_dim, 
                                nonlinearity, dropout, args):
        """Build architecture for image data."""
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.embed_dim = embed_dim
        self.args = args
        self.joint = args.joint if args else False
        self.num_time_head = getattr(args, 'num_time_head', 1)
        self.K = getattr(args, 'K', 300) if args else 300
        self.time_act = getattr(args, 'time_act', 'leakyrelu')
        
        # Time embedding
        self.time_embed = Layers.FourierSinusoidalTimeEncoder(output_dim=embed_dim, nfreq=embed_dim)
        
        if self.backbone == 'unet':
            # Build UNet-like architecture
            self._build_unet_architecture(in_channels, hidden_channels, embed_dim, nonlinearity, args)
        else:
            # Build ResNet/WideResNet architecture
            self._build_resnet_architecture(in_channels, hidden_channels, dropout, args)
    
    def _build_unet_architecture(self, in_channels, hidden_channels, embed_dim, nonlinearity, args):
        """Build UNet-like architecture."""
        # Initial input convolution (similar to the first layer in U-Net)
        self.init_conv = FusedConvBlock(in_channels, hidden_channels[0], None)
        
        # Dynamically build encoder and decoder
        self.encoder, self.decoder = self._build_unet_encoder_decoder()
        
        # Output heads
        if self.model_type == 'decomp_joint_score':
            # For decomposition models, build time head that outputs K components
            self.time_head = self._build_unet_time_head()
        else:
            # For other models, output single time score
            self.out_st = nn.Linear(hidden_channels[-1], 1)
        
        if self.joint:
            self.out_sx = nn.Conv2d(hidden_channels[0], in_channels, 1)
    
    def _build_unet_encoder_decoder(self):
        """Build UNet encoder and decoder."""
        encoder = nn.ModuleList()
        decoder = nn.ModuleList()
        
        # Encoder construction (downsampling path)
        in_ch = self.hidden_channels[0]
        for out_ch in self.hidden_channels[1:]:
            encoder.append(FusedConvBlock(in_ch, out_ch, None, stride=2))
            in_ch = out_ch

        reversed_ch = self.hidden_channels[::-1]
        for i in range(len(reversed_ch)-1):
            # Decoder layer parameters
            enc_ch = reversed_ch[i+1]  # Corresponding encoder channel
            dec_ch = reversed_ch[i+1]  # Target output channel
            
            decoder_block = nn.ModuleDict({
                # Upsampling layer (with channel adjustment)
                'up_conv': FusedConvBlock(
                    in_ch=reversed_ch[i],
                    out_ch=dec_ch,
                    embed_dim=None,
                    is_transpose=True,
                    stride=2
                ),
                # Skip connection adaptation layer
                'adapt_conv': nn.Conv2d(enc_ch, dec_ch, 1),
                # Feature fusion layer
                'merge_conv': FusedConvBlock(
                    in_ch=dec_ch*2,  # Upsampled channel + adapted encoder channel
                    out_ch=dec_ch,
                    embed_dim=self.embed_dim
                )
            })
            decoder.append(decoder_block)
        
        return encoder, decoder
    
    def _build_unet_time_head(self):
        """Build time head for UNet-based decomposition models."""
        layers = []
        ch = self.hidden_channels[-1]  
        conv_kwargs = {
            'in_channels': ch,
            'out_channels': ch,
            'kernel_size': 3,
            'padding': 1,
            'bias': False
        }
        for _ in range(self.num_time_head-1):
            layers += [ConvNormSilu(nn.Conv2d, num_groups=8, **conv_kwargs)]
        
        self.last_layer_time = nn.Conv2d(ch, self.K, 1)
        time_act = self._get_activation(self.time_act)
        layers += [
            self.last_layer_time,
            nn.AdaptiveAvgPool2d(1),
            time_act
        ]
        return nn.Sequential(*layers)
    
    def _build_resnet_architecture(self, in_channels, hidden_channels, dropout, args):
        """Build ResNet/WideResNet architecture."""
        # Build ResNet/WideResNet backbone
        if self.model_type in ["decomp_joint_score", "joint_score"]:
            out_dim = hidden_channels[-1]
        else:
            out_dim = 1
        if self.backbone == 'wideresnet28_10':
            self.backbone_net = Layers.WideResNet2810(out_dim=out_dim, dropRate=dropout)
        elif self.backbone in ['resnet18', 'resnet34', 'resnet50']:
            self.backbone_net = getattr(Layers, f'ResNet{self.backbone.replace("resnet", "")}')(out_dim=out_dim)
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
        
        backbone_out_dim = hidden_channels[-1]
        
        if self.model_type == 'decomp_joint_score':
            # Time head for decomposition - controlled by num_time_head
            time_head_layers = []
            for _ in range(getattr(args, 'num_time_head', 1) - 1):
                time_head_layers.extend([
                    nn.Linear(backbone_out_dim, backbone_out_dim),
                    nn.SiLU()
                ])
            time_head_layers.extend([
                nn.Linear(backbone_out_dim, self.K),
                nn.AdaptiveAvgPool2d(1),
                self._get_activation(self.time_act) if self.time_act != "identity" else nn.Identity()
            ])
            self.time_head = nn.Sequential(*time_head_layers)
        else:
            # Time head for regular joint score
            self.out_st = nn.Linear(backbone_out_dim, 1)
        
        # Data head for joint method - controlled by num_data_head and image dimensions
        if self.joint:
            # Get image dimensions from DATA_SHAPES based on args.data
            data_shape = getattr(args, 'data', None)
            if data_shape in DATA_SHAPES:
                img_channels, img_height, img_width = DATA_SHAPES[data_shape]
            else:
                # Default fallback to 32x32 if data not found in DATA_SHAPES
                img_channels, img_height, img_width = in_channels, 32, 32
            
            if self.model_type == 'decomp_joint_score':
                data_head_layers = []
                for _ in range(getattr(args, 'num_data_head', 1) - 1):
                    data_head_layers.extend([
                        nn.Linear(backbone_out_dim, backbone_out_dim),
                        nn.SiLU()
                    ])
                data_head_layers.append(nn.Linear(backbone_out_dim, img_channels * img_height * img_width))
                self.data_head = nn.Sequential(*data_head_layers)
            else:
                data_head_layers = []
                for _ in range(getattr(args, 'num_data_head', 1) - 1):
                    data_head_layers.extend([
                        nn.Linear(backbone_out_dim, backbone_out_dim),
                        nn.SiLU()
                    ])
                data_head_layers.append(nn.Linear(backbone_out_dim, img_channels * img_height * img_width))
                self.data_head = nn.Sequential(*data_head_layers)
    
    def _get_activation(self, nonlinearity):
        """Get activation function by name."""
        if nonlinearity in NONLINEARITIES.keys():
            if nonlinearity == "dyt":
                # Adjust input_dim based on context
                if self.model_type == 'decomp_joint_score':
                    input_dim = 4 if self.data_type == 'image' else 2
                else:
                    input_dim = 2
                return NONLINEARITIES[nonlinearity](num_features=self.K, input_dim=input_dim)
            return NONLINEARITIES[nonlinearity]
        else:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                act_name = self._detect_activation(module)
                INIT_METHODS.get(act_name, INIT_METHODS["identity"])(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.GroupNorm):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)
    
    def _detect_activation(self, layer):
        """Detect activation function from layer."""
        for _, child in itertools.islice(layer.named_children(), 3):
            if type(child) in ACTIVATION_TYPE_MAP:
                return ACTIVATION_TYPE_MAP[type(child)]
        return "identity"
    
    def reset_states(self):
        """Reset function evaluation counter."""
        self.nfe = 0
    
    def set_fk(self, f_k=None):
        """Set f_k values for decomposition models."""
        self.fk = f_k
    
    def get_fk_reg(self, x=None):
        """Get f_k values for decomposition models."""
        if not hasattr(self, 'fk'):
            return None

        if x is not None or self.fk is None:
            return self.compute_fk(x)
        else:
            return self.fk
    
    def _compute_fk_tabular(self, x):
        """Compute f_k features for tabular data."""
        # For tabular data, use shared layers and time head
        x = self.shared_layers(x)
        return self.time_head(x)  # Shape: (..., K)
    
    def _compute_fk_unet(self, x):
        """Compute f_k features for UNet-based models."""
        # For UNet, use init conv, encoder and time head
        x = self.init_conv(x, None)
        for block in self.encoder:
            x = block(x, None)
        return self.time_head(x).squeeze(-1).squeeze(-1)  # Shape: (..., K)
    
    def _compute_fk_resnet(self, x):
        """Compute f_k features for ResNet-based models."""
        # For ResNet, use backbone and time head
        # Ensure input has 3 channels
        if x.shape[1] == 1 and self.in_channels == 3:
            x = x.repeat(1, 3, 1, 1)
        
        features = self.backbone_net(x)
        
        # Process features through time head
        f_k = self.time_head(features)
        if f_k.dim() == 4:
            f_k = f_k.squeeze(-1).squeeze(-1)
        elif f_k.dim() == 3:
            f_k = f_k.squeeze(-1)
        return f_k
    
    def compute_fk(self, x):
        """Compute f_k features for decomposition models."""
        return self.compute_fk_impl(x)
    
    def compute_weighted_sum(self, x, f_k=None):
        """Compute the weighted sum for decomposition models.""" 
        if f_k is None:
            f_k = self.compute_fk(x)
        weighted_sum = torch.sum(f_k * self.Pk.integrals, dim=-1, keepdim=True)
        return weighted_sum.squeeze()
    
    def time_gradient(self, t, x, f_k=None):
        """Compute the gradient with respect to time for decomposition models."""
        if f_k is None:
            f_k = self.compute_fk(x)
        g_k = self.Pk.grad_function(t)
        return torch.sum(f_k * g_k, dim=-1, keepdim=True)
    
    def forward(self, *inputs, cond=None, joint=True):
        """
        Forward pass for the unified model.
        
        For time-dependent models (joint_score, decomp_joint_score):
        - inputs[0]: time t
        - inputs[1]: data x
        For density ratio models:
        - inputs[0]: data x
        """
        return self.forward_impl(*inputs, cond=cond, joint=joint)
    
    def _forward_time_dependent(self, t, x, cond=None, joint=True):
        """Forward pass for time-dependent models."""
        return self._forward_time_dependent_impl(t, x, cond, joint)

    def _compute_s_t_tabular_decomp(self, t, x, time_emb):
        """Compute s_t for tabular decomposition models."""
        f_k = self.time_head(x)  # Shape: (..., K)
        g_k = self.Pk(t)  # Shape: (..., K)
        s_t = torch.sum(f_k * g_k, dim=-1, keepdim=True)  # Shape: (..., 1)

        # Clamp the values to avoid extreme values leading to NaN in gradients
        f_k = torch.clamp(f_k, min=-1e6, max=1e6)
        g_k = torch.sign(g_k) * torch.clamp(torch.abs(g_k), min=1e-8)

        self.set_fk(f_k)

        if torch.isnan(s_t).any().item():
            print("f_k:", f_k.mean(dim=0), "\n g_k:", g_k.mean(dim=0), "\n s_t: ", s_t.mean(dim=0))
            raise ValueError("NaN detected in time score computation")
        return s_t

    def _compute_s_t_tabular_joint(self, t, x, time_emb):
        """Compute s_t for tabular joint-score models."""
        s_t = x
        for layer in self.time_head:
            if hasattr(layer, 'forward'):
                if hasattr(layer, 'embed_dim'):
                    s_t = layer(s_t, time_emb=time_emb)
                else:
                    s_t = layer(s_t)
            else:
                s_t = layer(s_t)
        return s_t

    def _compute_s_t_unet_decomp(self, t, features, t_emb):
        """Compute s_t for UNet decomposition models."""
        f_k = self.time_head(features).squeeze(-1).squeeze(-1)  # Shape: (batch_size, K)
        g_k = self.Pk(t)   # Shape: (..., K)
        s_t = torch.sum(f_k * g_k, dim=-1, keepdim=True)  # Shape: (..., 1)
        self.set_fk(f_k)

        if torch.isnan(s_t).any().item():
            raise ValueError("NaN detected in time score computation")
        return s_t

    def _compute_s_t_unet_joint(self, t, features, t_emb):
        """Compute s_t for UNet joint-score models."""
        return self.out_st(features.mean(dim=(2, 3))).view(-1, 1)

    def _compute_s_t_resnet_decomp(self, t, features, t_emb):
        """Compute s_t for ResNet decomposition models."""
        f_k = self.time_head(features)
        if f_k.dim() == 4:
            f_k = f_k.squeeze(-1).squeeze(-1)
        elif f_k.dim() == 3:
            f_k = f_k.squeeze(-1)
        g_k = self.Pk(t)   # Shape: (..., K)
        s_t = torch.sum(f_k * g_k, dim=-1, keepdim=True)  # Shape: (..., 1)
        self.set_fk(f_k)

        if torch.isnan(s_t).any().item():
            raise ValueError("NaN detected in time score computation")
        return s_t

    def _compute_s_t_resnet_joint(self, t, features, t_emb):
        """Compute s_t for ResNet joint-score models."""
        return self.out_st(features).view(-1, 1)  # [B, 1]
    
    def _forward_tabular_time_dependent(self, t, x, cond=None, joint=True):
        """Forward pass for tabular time-dependent models."""
        self.nfe += 1

        time_emb = self.time_emb(t)  # [B, timeemb_outdim] 
        if cond is not None: 
            rel_time_emb = self.time_emb(t - cond)
            # rel_time_emb = self.delta_emb(t - cond)
            concat_emb = torch.cat([time_emb, rel_time_emb], dim=-1)
            time_emb = self.time_fusion(concat_emb) if hasattr(self, 'time_fusion') else time_emb
            # time_emb += self.time_emb(t - cond)
        
        # Process through shared layers
        for layer in self.shared_layers:
            if hasattr(layer, 'forward'):  # Check if layer expects time embedding
                if hasattr(layer, 'embed_dim'):
                    x = layer(x, time_emb=time_emb)
                else:
                    x = layer(x)
            else:
                x = layer(x)

        s_t = self._compute_s_t_impl(t, x, time_emb)
        
        if self.joint and joint:
            s_x = x
            for layer in self.data_head:
                if hasattr(layer, 'embed_dim'):
                    s_x = layer(s_x, time_emb=time_emb)
                else:
                    s_x = layer(s_x)
            return s_x, s_t  
        return s_t
    
    def _forward_unet_time_dependent(self, t, x, cond=None, joint=True):
        """Forward pass for UNet-based time-dependent models."""
        self.nfe += 1
        t_emb = self.time_embed(t).squeeze()
        if cond is not None:
            rel_temb = self.time_embed(t - cond)
            # Apply time fusion if available
            if hasattr(self, 'time_fusion'):
                concat_emb = torch.cat([t_emb, rel_temb], dim=-1)
                t_emb = self.time_fusion(concat_emb)
        
        x = self.init_conv(x, t_emb)
        skips = [x]  # Store encoder features
        
        # Encoding process (downsampling)
        for block in self.encoder:
            x = block(x, t_emb)
            skips.append(x)
        
        s_t = self._compute_s_t_impl(t, x, t_emb)
        
        if self.joint and joint:
            # Decoding process (upsampling)
            for i, dec_block in enumerate(self.decoder):
                # Upsampling operation
                x = dec_block['up_conv'](x, t_emb)
                
                # Adapt encoder features to current channel
                enc_feat = skips[-(i+2)]  # Skip the latest feature
                enc_feat = dec_block['adapt_conv'](enc_feat)
                
                # Feature fusion
                x = torch.cat([x, enc_feat], dim=1)
                x = dec_block['merge_conv'](x, t_emb)
            
            s_x = self.out_sx(x)
            return s_x, s_t
        return s_t
    
    def _forward_resnet_time_dependent(self, t, x, cond=None, joint=True):
        """Forward pass for ResNet-based time-dependent models."""
        self.nfe += 1
        
        # Get time embedding
        t_emb = self.time_embed(t).squeeze()
        if cond is not None:
            rel_temb = self.time_embed(t - cond)
            # Apply time fusion if available
            if hasattr(self, 'time_fusion'):
                concat_emb = torch.cat([t_emb, rel_temb], dim=-1)
                t_emb = self.time_fusion(concat_emb)
        
        # Process through backbone
        # Ensure input has 3 channels
        if x.shape[1] == 1 and self.in_channels == 3:
            x = x.repeat(1, 3, 1, 1)
        
        features = self.backbone_net(x)  # [B, backbone_out_dim]
        
        s_t = self._compute_s_t_impl(t, features, t_emb)
        
        if self.joint and joint:
            # Data score (image reconstruction)
            s_x = self.data_head(features)
            # Use pre-computed image dimensions
            s_x = s_x.view(x.shape[0], self._img_channels, self._img_height, self._img_width)
            return s_x, s_t
        return s_t
    
    def _forward_density_ratio_tabular(self, x, cond=None, joint=True):
        """Forward pass for tabular density-ratio models."""
        for layer in self.shared_layers:
            x = layer(x)

        for layer in self.time_head:
            x = layer(x)

        return x

    def _forward_density_ratio_image_resnet(self, x, cond=None, joint=True):
        """Forward pass for ResNet/WideResNet density-ratio models."""
        if x.shape[1] == 1 and self.in_channels == 3:
            x = x.repeat(1, 3, 1, 1)

        logits = self.backbone_net(x)
        if logits.size(-1) == 1:
            return logits.squeeze(-1)
        return logits

    def _forward_density_ratio_image_unet(self, x, cond=None, joint=True):
        """Forward pass for UNet density-ratio models."""
        x = self.init_conv(x, None)
        for block in self.encoder:
            x = block(x, None)

        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.out_st(x) if hasattr(self, 'out_st') else x
        return x.squeeze()


# ==================== Convenience Classes for Backward Compatibility ====================

# Non-image models
class JointScoreModel(UnifiedModel):
    def __init__(self, input_dim, hidden_dims, embed_dim=256, nonlinearity="leakyrelu", 
                 args=None, norm=False, use_spectral_norm=False, dynamic=False):
        super().__init__(
            model_type='joint_score',
            data_type='tabular',
            backbone='mlp',
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            embed_dim=embed_dim,
            nonlinearity=nonlinearity,
            args=args,
            norm=norm,
            use_spectral_norm=use_spectral_norm,
            dynamic=dynamic
        )

class NeuralDensityRatioModel(UnifiedModel):
    def __init__(self, input_dim, hidden_dims, nonlinearity="leakyrelu", 
                 args=None, norm=False, use_spectral_norm=False, dynamic=False):
        super().__init__(
            model_type='density_ratio',
            data_type='tabular',
            backbone='mlp',
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            nonlinearity=nonlinearity,
            args=args,
            norm=norm,
            use_spectral_norm=use_spectral_norm,
            dynamic=dynamic
        )

# Image models
class ImageJointScoreModel(UnifiedModel):
    def __init__(self, args, in_channels, hidden_channels, embed_dim=128, nonlinearity="leakyrelu", backbone='unet', dropout=0.0):
        super().__init__(
            model_type='joint_score',
            data_type='image',
            backbone=backbone,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            embed_dim=embed_dim,
            nonlinearity=nonlinearity,
            args=args,
            dropout=dropout
        )

class ImageDecompJointScoreModel(UnifiedModel):
    def __init__(self, args, in_channels, hidden_channels, embed_dim=128, K=300, 
                 nonlinearity="leakyrelu", polynomials="gaussrbf", noise_type="zero"):
        super().__init__(
            model_type='decomp_joint_score',
            data_type='image',
            backbone='unet',
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            K=K,
            embed_dim=embed_dim,
            nonlinearity=nonlinearity,
            polynomials=polynomials,
            noise_type=noise_type,
            args=args
        )

class ImageJointScoreModelResNet(UnifiedModel):
    def __init__(self, args, in_channels, hidden_channels, embed_dim=128, nonlinearity="leakyrelu",
                 backbone='resnet18', dropout=0.3):
        super().__init__(
            model_type='joint_score',
            data_type='image',
            backbone=backbone,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            embed_dim=embed_dim,
            nonlinearity=nonlinearity,
            args=args,
            dropout=dropout
        )

class ImageDecompJointScoreModelResNet(UnifiedModel):
    def __init__(self, args, in_channels, hidden_channels, embed_dim=128, K=300, 
                 nonlinearity="leakyrelu", polynomials="gaussrbf", noise_type="zero",
                 backbone='resnet18', dropout=0.3):
        super().__init__(
            model_type='decomp_joint_score',
            data_type='image',
            backbone=backbone,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            K=K,
            embed_dim=embed_dim,
            nonlinearity=nonlinearity,
            polynomials=polynomials,
            noise_type=noise_type,
            args=args,
            dropout=dropout
        )

class ResNetDensityRatioModel(UnifiedModel):
    def __init__(self, in_channels=3, num_classes=1, backbone='resnet18', dropout=0.3):
        # Create dummy args to satisfy parent constructor
        class DummyArgs:
            def __init__(self):
                self.joint = False

        super().__init__(
            model_type='density_ratio',
            data_type='image',
            backbone=backbone,
            in_channels=in_channels,
            hidden_channels=[64, 128, 256, 512],  # Placeholder
            embed_dim=128,  # Placeholder
            args=DummyArgs(),
            dropout=dropout
        )
    
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
            'bias': False  # GroupNorm has bias, so no need for conv bias
        }
        conv2_kwargs = {
            'in_channels': in_ch,
            'out_channels': out_ch,
            'kernel_size': 3,
            'stride': 1,    # (fixed stride=1 to maintain size)
            'padding': 1,
            'bias': False
        }
        
        # Only transpose convolution requires output padding
        if is_transpose:
            conv1_kwargs['output_padding'] = stride // 2  # Ensure precise size alignment

        self.layers = nn.ModuleList([
            ConvNormSilu(conv_cls, num_groups=num_groups, **conv1_kwargs),
            ConvNormSilu(conv_cls, num_groups=num_groups, **conv2_kwargs),
        ])

        # Time conditioning projection
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

        # self.diffeq = diffeq_layers.wrappers.diffeq_wrapper(diffeq)
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

        # increment num evals
        self._num_evals += 1

        # convert to tensor
        # t = torch.tensor(t).type_as(y)
        t = t.clone().detach().to(y)
        batchsize = y.shape[0]

        # Sample and fix the noise.
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
            # Hack for 2D data to use brute force divergence computation.
            if not self.training and dy.view(dy.shape[0], -1).shape[1] == 2:
                divergence = divergence_bf(dy, y).view(batchsize, 1)
            else:
                divergence = self.divergence_fn(dy, y, e=self._e).view(batchsize, 1)
        if self.residual:
            dy = dy - y
            divergence -= torch.ones_like(divergence) * torch.tensor(np.prod(y.shape[1:]), dtype=torch.float32
                                                                     ).to(divergence)
        return tuple([dy, -divergence] + [torch.zeros_like(s_).requires_grad_(True) for s_ in states[2:]])

class cnf_net(torch.nn.Module):
    def __init__(self, args, hidden_dims, device='cuda', func=None):
        super(cnf_net, self).__init__()
        self.activation = args.nonlinearity
        self.register_buffer("sqrt_end_time", torch.sqrt(torch.tensor(1.)))
        dim = args.d
        if func is None:
            # odefunc = myodefunc(dim, activation=self.activation, hidden_dims=hidden_dims).to(device)
            diffeq = JointScoreModel(hidden_dims=hidden_dims,
                            input_shape=(dim,),
                            strides=None,
                            conv=False,
                            layer_type=args.layer_type,
                            nonlinearity=args.nonlinearity,)
            odefunc = ODEfunc(diffeq=diffeq, divergence_fn=args.divergence_fn, residual=args.residual, rademacher=args.rademacher)
            odefunc.before_odeint()
        else:
            odefunc = func
        self.time_deriv_func = odefunc  # 可以得到eq.4中被积函数的-Tr项
        self.odefunc = self.time_deriv_func
        self.NFE = 0
        self.solver = args.solver
        self.rtol = args.rtol
        self.atol = args.atol
        
    def save_state(self, fn='state.tar'):
        """Save model state."""
        torch.save(self.state_dict(), fn)

    def load_state(self, fn='state.tar'):
        """Load model state."""
        self.load_state_dict(torch.load(fn, map_location='cpu'))

    def forward(self, z, delta_logpz=None, integration_times=None, reverse=False):
        device = z.device
        if delta_logpz is None:
            delta_logpz = torch.zeros(z.shape[0], 1).to(device)
        if integration_times is None:
            integration_times = torch.tensor([0.0, 1.0]).to(z) 
        if reverse:
            integration_times = _flip(integration_times, 0)

        # Integrate. This is the call to torchdiffeq.
        self.odefunc.before_odeint()
        state = odeint(
            self.time_deriv_func,  # Calculates time derivatives.
            (z, delta_logpz),  # Values to update. 
            integration_times,  # When to evaluate.
            method=self.solver,  # Runge-Kutta
            atol=self.atol,  # Absolute tolerance
            rtol=self.rtol,  # Relative tolerance
        )
        self.NFE = self.odefunc._num_evals

        if len(integration_times) == 2:
            state = tuple(s[1] for s in state)
        z, delta_logpz = state
        return z, delta_logpz

# In[] Kernel-based density ratio estimation
class KernelDensityRatioModel(nn.Module):
    """
    Parametric density ratio estimation model.
    """
    def __init__(self, args, sde=None, method='kulsif'):
        super().__init__()
        self.args = args
        self.method = method
        # Initial regularization parameter for KuLSIF (will be overridden by grid search)
        self.reg_param = getattr(args, 'reg_param', 1.0) 
        self.sde = sde
        # Increase the upper limit on the number of training samples
        self.max_samples = getattr(args, 'max_parametric_samples', 2000) 
        
        self.is_trained = False
        self.training_data = {'qx': [], 'px': []}
        self.epoch_count = 0
        
        # Model parameters
        self.alpha = None
        self.training_points = None
        self.rbf_gamma_value = None
        
        # Data normalization parameters
        self.data_mean = None
        self.data_std = None
        
        self.dummy_param = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        
    def forward(self, x):
        """Predict density ratio r(x) = p1(x)/p0(x)."""
        if not self.is_trained or self.alpha is None:
            return torch.ones(x.shape[0], device=x.device)
        
        x_np = x.detach().cpu().numpy()
        
        if self.data_mean is not None:
            x_np = (x_np - self.data_mean) / (self.data_std + 1e-8)
        
        predictions = self._predict_batchwise(x_np)
        predictions = np.clip(predictions, 1e-6, 1e4)
        
        result = torch.from_numpy(predictions).float().to(x.device)
        
        if not hasattr(self, '_debug_printed'):
            print(f"[Forward] Predictions: min={predictions.min():.4f}, max={predictions.max():.4f}, mean={predictions.mean():.4f}")
            self._debug_printed = True
        
        return result
    
    def collect_data(self, qx, px):
        """
        Collect training data – continuously accumulate, but limit total number of batches.
        """
        qx_np = qx.detach().cpu().numpy()
        px_np = px.detach().cpu().numpy()
        
        max_per_batch = 300
        if len(qx_np) > max_per_batch:
            idx = np.random.choice(len(qx_np), max_per_batch, replace=False)
            qx_np = qx_np[idx]
        if len(px_np) > max_per_batch:
            idx = np.random.choice(len(px_np), max_per_batch, replace=False)
            px_np = px_np[idx]

        self.training_data['qx'].append(qx_np)
        self.training_data['px'].append(px_np)

        MAX_TOTAL_BATCHES = 20  # Limit to at most 20 batches
        if len(self.training_data['qx']) > MAX_TOTAL_BATCHES:
            self.training_data['qx'] = self.training_data['qx'][-MAX_TOTAL_BATCHES:]
            self.training_data['px'] = self.training_data['px'][-MAX_TOTAL_BATCHES:]

        total = sum(len(arr) for arr in self.training_data['qx']) + \
                sum(len(arr) for arr in self.training_data['px'])
        
        if len(self.training_data['qx']) == 1:
            print(f"Collected data: {len(qx_np)} p0, {len(px_np)} p1 samples. Total: {total}")
    
    def train_model(self):
        """Train the density ratio model – perform KuLSIF parameter search only on first training."""
        self.epoch_count += 1
        
        if len(self.training_data['qx']) == 0 or len(self.training_data['px']) == 0:
            print(f"Warning: No training data available at epoch {self.epoch_count}")
            return
        
        all_qx = np.vstack(self.training_data['qx'])
        all_px = np.vstack(self.training_data['px'])
        
        print(f"\n{'='*60}")
        print(f"Training Epoch {self.epoch_count} - Method: {self.method}")
        
        # 1. Data normalization and subsampling (unchanged)
        X_all = np.vstack([all_qx, all_px])
        self.data_mean = np.mean(X_all, axis=0)
        self.data_std = np.std(X_all, axis=0) + 1e-8
        
        all_qx = (all_qx - self.data_mean) / self.data_std
        all_px = (all_px - self.data_mean) / self.data_std
        
        n_target = min(self.max_samples, len(all_qx), len(all_px))
        
        # Ensure equal number of p0 and p1 samples
        np.random.shuffle(all_qx)
        np.random.shuffle(all_px)
        all_qx = all_qx[:n_target]
        all_px = all_px[:n_target]
        
        print(f"Training with {len(all_qx)} p0 and {len(all_px)} p1 samples")

        # 2. Determine RBF gamma (run only once)
        if self.rbf_gamma_value is None:
            self.rbf_gamma_value = self._compute_rbf_gamma(all_qx, all_px)
        
        # 3. KuLSIF parameter grid search (run only once and only if method is 'kulsif')
        if self.method == 'kulsif' and self.epoch_count == 1:
            self._perform_kulsif_grid_search(all_qx, all_px)
            # After search, self.reg_param is updated to the best value

        print(f"RBF gamma: {self.rbf_gamma_value:.6f}, Regularization: {self.reg_param:.4f}")
        
        try:
            if self.method == 'kulsif':
                self._train_kulsif(all_qx, all_px, self.reg_param, self.rbf_gamma_value)
            elif self.method == 'kliep':
                self._train_kliep(all_qx, all_px, self.rbf_gamma_value)
            
            if self.alpha is not None:
                # Validate on training set
                test_qx = self._predict_batchwise(all_qx[:100])
                test_px = self._predict_batchwise(all_px[:100])
                
                print(f"Ratios on p0: mean={test_qx.mean():.4f} (should be approx 1)")
                print(f"Ratios on p1: mean={test_px.mean():.4f} (should be >1)")
                print(f"Alpha: min={self.alpha.min():.4f}, max={self.alpha.max():.4f}, "
                      f"mean={self.alpha.mean():.4f}, norm={np.linalg.norm(self.alpha):.4f}")
                
                self.is_trained = True
                print("Training successful!")
                
                if hasattr(self, '_debug_printed'):
                    delattr(self, '_debug_printed')
            else:
                print("Training failed: alpha is None")
                
        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"{'='*60}\n")
    
    def _compute_rbf_gamma(self, qx, px):
        """Compute RBF gamma using the median heuristic: 1 / median^2."""
        n_samples = min(500, len(qx), len(px))
        qx_sample = qx[np.random.choice(len(qx), n_samples, replace=False)]
        px_sample = px[np.random.choice(len(px), n_samples, replace=False)]
        X = np.vstack([qx_sample, px_sample])
        
        distances = pdist(X)
        median_dist = np.median(distances)
        
        if median_dist < 1e-6:
            return 1.0
        
        # Standard heuristic: 1 / (median_dist^2)
        gamma = 1.0 / (median_dist ** 2) 
        gamma = np.clip(gamma, 0.001, 10.0)  # Cap gamma to avoid excessively large values
        
        return gamma
        
    def _perform_kulsif_grid_search(self, all_qx, all_px):
        """
        Perform parameter grid search and single-fold cross-validation for KuLSIF.
        Search over lambda (reg_param).
        """
        print("\n--- Starting KuLSIF Parameter Grid Search ---")

        # 1. Split into training and validation sets (one-fold CV)
        X = np.vstack([all_qx, all_px])
        # Target labels (y=1: p1, y=0: p0)
        y = np.hstack([np.zeros(len(all_qx)), np.ones(len(all_px))]) 
        
        # Split data into 80% training and 20% validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Separate p0 and p1 samples from X_train
        qx_train = X_train[y_train == 0]
        px_train = X_train[y_train == 1]
        
        # Define centers for kernel computation
        # Alpha is always computed relative to X_train, so centers must be X_train
        X_centers = X_train
        
        # 2. Define search grid
        lambda_candidates = 10.0 ** (-np.arange(start=-3.0, step=1.0, stop=5.0)) 
        
        best_score = -np.inf
        best_lambda = self.reg_param  # initial value
        gamma = self.rbf_gamma_value

        # 3. Evaluate each candidate
        for reg_param in lambda_candidates:
            try:
                # Temporarily train alpha using X_centers (X_train)
                alpha, cond_num = self._train_kulsif_temp(qx_train, px_train, gamma, reg_param)
                if alpha is None: continue

                # Evaluation metric: maximize E_{p1}[r(x)]
                # Centers must be X_centers (X_train)
                K_val = rbf_kernel(X_val[y_val==1], X_centers, gamma=gamma)
                ratios_p1_val = K_val @ alpha
                
                # Use mean ratio on validation p1 samples as the score
                val_score = np.mean(ratios_p1_val) 

                if val_score > best_score:
                    best_score = val_score
                    best_lambda = reg_param
                    
                print(f"  Lambda={reg_param:.4e}: R(p1)_mean={val_score:.4f}, Cond={cond_num:.2e}")

            except Exception as e:
                import traceback
                print(f"  Lambda={reg_param:.4e}: Failed (Error: {type(e).__name__})")
                continue

        self.reg_param = best_lambda
        print(f"--- Search Complete: Best Lambda={best_lambda:.4e}, R(p1)_mean={best_score:.4f} ---")

    def _train_kulsif_temp(self, qx, px, gamma, reg_param):
        """Temporary KuLSIF training function used during grid search."""
        n0, n1 = len(qx), len(px)
        # X serves as centers: qx + px
        X = np.vstack([qx, px]) 
        
        K = rbf_kernel(X, X, gamma=gamma)
        K_p0 = K[:n0, :]
        
        # Target vector h uses kernel mean over p1 samples
        h = np.mean(K[n0:, :], axis=0) 
        
        # Hessian matrix
        H = K_p0.T @ K_p0 / n0
        
        cond_num = np.linalg.cond(H)
        
        H_reg = H + reg_param * np.eye(len(X))
        
        try:
            alpha = np.linalg.solve(H_reg, h)
        except np.linalg.LinAlgError:
            # Fallback to least squares if direct solve fails
            alpha = np.linalg.lstsq(H_reg, h, rcond=1e-5)[0]
        
        alpha = np.maximum(alpha, 0)
        
        # Normalize so that E_{p0}[r(x)] = 1
        mean_ratio_p0 = np.mean(K_p0 @ alpha)
        if mean_ratio_p0 > 1e-6:
            alpha = alpha / mean_ratio_p0
            return alpha, cond_num
        
        return None, cond_num  # Training failed

    def _train_kulsif(self, qx, px, reg_param, gamma):
        """KuLSIF method – train final model using best parameters."""
        n0, n1 = len(qx), len(px)
        self.training_points = X = np.vstack([qx, px])
        
        K = rbf_kernel(X, X, gamma=gamma)
        K_p0 = K[:n0, :]
        
        h = np.mean(K[n0:, :], axis=0)
        H = K_p0.T @ K_p0 / n0
        
        cond_num = np.linalg.cond(H)
        print(f"H condition number (Final): {cond_num:.2e}")
        
        final_reg = reg_param
        
        # If condition number is too high, enforce stronger regularization for stability
        if cond_num > 1e12:  # 1e12 is a threshold for severe ill-conditioning
            safe_reg = max(reg_param, 100.0)
            if cond_num > 1e14:
                safe_reg = max(safe_reg, 500.0)
            
            final_reg = safe_reg
            print(f"Applying strong regularization due to cond_num. Final Reg={final_reg:.4f}")
        
        H_reg = H + final_reg * np.eye(len(X))
        
        try:
            self.alpha = np.linalg.solve(H_reg, h)
        except np.linalg.LinAlgError:
            self.alpha = np.linalg.lstsq(H_reg, h, rcond=1e-5)[0]
        
        self.alpha = np.maximum(self.alpha, 0)
        
        mean_ratio_p0 = np.mean(K_p0 @ self.alpha)
        if mean_ratio_p0 > 1e-6:
            self.alpha = self.alpha / mean_ratio_p0
            print(f"Normalized alpha by factor {mean_ratio_p0:.4f}")
        else:
            print("Warning: Final KuLSIF normalization failed.")
            self.alpha = None
    
    def _train_kliep(self, qx, px, gamma):
        """KLIEP method – train using fixed gamma."""
        n0, n1 = len(qx), len(px)
        self.training_points = X = np.vstack([qx, px])
        
        K = rbf_kernel(X, X, gamma=gamma)
        
        K_p0 = K[:n0, :]
        K_p1_train = K[n0:, :] 
        
        alpha = np.ones(K.shape[1]) / K.shape[1]
        
        # Increase iterations to ensure convergence
        for iteration in range(1000): 
            r_p1 = K_p1_train @ alpha
            r_p1 = np.maximum(r_p1, 1e-10)
            
            grad = np.mean(K_p1_train / r_p1[:, np.newaxis], axis=0)
            alpha_new = alpha * grad
            
            norm = np.mean(K_p0 @ alpha_new)
            if norm > 1e-8:
                alpha_new = alpha_new / norm
            else:
                break
            
            if np.max(np.abs(alpha_new - alpha)) < 1e-6:
                print(f"KLIEP converged at iteration {iteration}")
                break
            
            alpha = alpha_new
            
            if iteration % 200 == 0:
                r_p0_mean = np.mean(K_p0 @ alpha)
                r_p1_mean = np.mean(K_p1_train @ alpha)
                print(f"KLIEP Iter {iteration}: R(p0)={r_p0_mean:.4f}, R(p1)={r_p1_mean:.4f}, alpha_norm={np.linalg.norm(alpha):.4f}")
        
        self.alpha = alpha
    
    def _predict_batchwise(self, x, batch_size=1000):
        """Make predictions in batches."""
        if self.alpha is None or self.training_points is None:
            return np.ones(len(x))
        
        n = len(x)
        predictions = np.zeros(n)
        
        gamma = self.rbf_gamma_value  # Use gamma determined during training
        
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            x_batch = x[i:end_i]
            
            K = rbf_kernel(x_batch, self.training_points, gamma=gamma)
            predictions[i:end_i] = K @ self.alpha
        
        return predictions
