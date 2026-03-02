# -*- coding: utf-8 -*-
import itertools
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import layers.layers as Layers
import polynomials.polynomials as Pk
from pl_data import DATA_SHAPES
from layers.activations import NONLINEARITIES, INIT_METHODS, ACTIVATION_TYPE_MAP, ConvNormSilu
from .blocks import FusedConvBlock
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

