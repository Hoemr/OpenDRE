# -*- coding: utf-8 -*-
"""Image-data neural network models: UNet/ResNet joint score and density ratio.

Compared to the previous unified ``UnifiedModel``, these classes only focus on
image-based architectures (UNet / ResNet / WideResNet) and their forward logic.
Tabular logic lives in ``tabular_models.py`` so that each task family is
encapsulated in its own module.
"""

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import nn_models.layers.layers as Layers
from pl_data import DATA_SHAPES
from nn_models.layers.activations import (
    NONLINEARITIES,
    INIT_METHODS,
    ACTIVATION_TYPE_MAP,
    ConvNormSilu,
)
from nn_models.layers.blocks import FusedConvBlock


class UNetDataHead(nn.Module):
    """Data head for UNet-based models with optional time conditioning.

    This module supports multiple FusedConvBlock layers that can condition on
    time embedding, followed by a final 1x1 convolution.
    """

    def __init__(self, in_channels, out_channels, embed_dim=None, num_layers=1, num_groups=8):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(FusedConvBlock(in_channels, in_channels, embed_dim=embed_dim, num_groups=num_groups))
        self.final_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, t_emb=None):
        for layer in self.layers:
            x = layer(x, t_emb)
        return self.final_conv(x)


class _BaseImageModel(nn.Module):
    """Shared utilities for image models (EDM preconditioning, init, etc.)."""

    def __init__(self, args=None, path=None):
        super().__init__()
        self.args = args
        self.path = path
        self.joint = args.joint if args is not None else False

        # EDM configuration (kept consistent with UnifiedModel)
        self.use_edm_precond = getattr(args, "use_edm_precond", False)
        self.edm_sigma_data = getattr(args, "edm_sigma_data", 1.0)
        self.edm_sigma_min = getattr(args, "edm_sigma_min", 1e-3)
        self.edm_use_noise_embed = getattr(args, "edm_use_noise_embed", False)
        self.edm_debug = getattr(args, "edm_debug", False)
        self.register_buffer("edm_v_src", None)
        self.register_buffer("edm_v_dst", None)

        self.noise_embed = None
        self.nfe = 0  # number of function evaluations

    # --------------------
    # Initialization and activations
    # --------------------
    def _get_activation(self, nonlinearity):
        """Return activation function by name (mirrors UnifiedModel._get_activation)."""
        if nonlinearity in NONLINEARITIES.keys():
            if nonlinearity == "dyt":
                input_dim = 2
                K = getattr(self, "K", 1)
                return NONLINEARITIES[nonlinearity](num_features=K, input_dim=input_dim)
            return NONLINEARITIES[nonlinearity]
        else:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")

    def _detect_activation(self, layer):
        """Detect activation type from submodules (aligned with UnifiedModel)."""
        for _, child in itertools.islice(layer.named_children(), 3):
            if type(child) in ACTIVATION_TYPE_MAP:
                return ACTIVATION_TYPE_MAP[type(child)]
        return "identity"

    def _initialize_weights(self):
        """Initialize network weights (copied from UnifiedModel for consistency)."""
        for _, module in self.named_modules():
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

    # --------------------
    # EDM preconditioning
    # --------------------
    def set_edm_endpoint_stats(self, v_src, v_dst):
        """Set EDM endpoint statistics (same API as UnifiedModel)."""
        dev = next(self.parameters()).device
        self.register_buffer("edm_v_src", v_src.to(dev))
        self.register_buffer("edm_v_dst", v_dst.to(dev))

    def _edm_precond_input(self, t, x):
        """EDM input scaling (copied from UnifiedModel._edm_precond_input)."""
        if not self.use_edm_precond or self.path is None:
            return x, None

        s_t2 = self.path.total_scale_squared_t_only(t)
        s_t = torch.sqrt(s_t2.clamp(min=1e-10)).clamp(min=self.edm_sigma_min)
        c_in = 1.0 / torch.sqrt(self.edm_sigma_data ** 2 + s_t ** 2)
        if x.dim() == 4:
            c_in = c_in.view(-1, 1, 1, 1)
        elif x.dim() == 2:
            c_in = c_in.view(-1, 1)
        x_scaled = c_in * x
        c_noise = None
        if self.edm_use_noise_embed and self.noise_embed is not None:
            s_t_scalar = s_t
            if s_t.dim() == 2 and s_t.shape[1] > 1:
                s_t_scalar = torch.sqrt((s_t ** 2).mean(dim=1, keepdim=True))
            c_noise = 0.25 * torch.log(s_t_scalar.clamp(min=1e-10))
        if self.edm_debug:
            with torch.no_grad():
                path_type = "bridge" if getattr(self.path, "bridge", False) else "default"
                noise_embed_on = self.edm_use_noise_embed and self.noise_embed is not None
                st_mean, st_std = s_t.float().mean().item(), s_t.float().std().item()
                st_min, st_max = s_t.float().min().item(), s_t.float().max().item()
                x_mean, x_std = x.float().mean().item(), x.float().std().item()
                xs_mean, xs_std = x_scaled.float().mean().item(), x_scaled.float().std().item()
                import logging

                logging.getLogger(__name__).info(
                    f"EDM_precond: path={path_type} noise_embed={noise_embed_on} | "
                    f"s_t mean={st_mean:.4f} std={st_std:.4f} min={st_min:.4f} max={st_max:.4f} | "
                    f"x_t raw mean={x_mean:.4f} std={x_std:.4f} | x_t^in mean={xs_mean:.4f} std={xs_std:.4f}"
                )
        return x_scaled, c_noise

    # --------------------
    # Misc
    # --------------------
    def reset_states(self):
        self.nfe = 0


class _BaseImageTimeDependentModel(_BaseImageModel):
    """Base class for time-dependent image models (joint_score)."""

    def _build_time_embedding(self, embed_dim):
        """Time embedding used by image models (Fourier encoder)."""
        if self.joint:
            self.time_embed = Layers.FourierSinusoidalTimeEncoder(
                output_dim=embed_dim, nfreq=embed_dim
            )
        else:
            self.time_embed = None

    def _build_unet_architecture(self, in_channels, hidden_channels, embed_dim, num_time_head):
        """Build UNet-like encoder/decoder plus heads."""
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.embed_dim = embed_dim
        self.num_time_head = num_time_head

        # Initial conv
        self.init_conv = FusedConvBlock(in_channels, hidden_channels[0], None)

        # Encoder
        encoder = nn.ModuleList()
        in_ch = hidden_channels[0]
        for out_ch in hidden_channels[1:]:
            encoder.append(FusedConvBlock(in_ch, out_ch, None, stride=2))
            in_ch = out_ch
        self.encoder = encoder

        # Decoder
        decoder = nn.ModuleList()
        reversed_ch = hidden_channels[::-1]
        for i in range(len(reversed_ch) - 1):
            enc_ch = reversed_ch[i + 1]
            dec_ch = reversed_ch[i + 1]
            decoder_block = nn.ModuleDict(
                {
                    "up_conv": FusedConvBlock(
                        in_ch=reversed_ch[i],
                        out_ch=dec_ch,
                        embed_dim=None,
                        is_transpose=True,
                        stride=2,
                    ),
                    "adapt_conv": nn.Conv2d(enc_ch, dec_ch, 1),
                    "merge_conv": FusedConvBlock(
                        in_ch=dec_ch * 2,
                        out_ch=dec_ch,
                        embed_dim=self.embed_dim,
                    ),
                }
            )
            decoder.append(decoder_block)
        self.decoder = decoder

    def _build_unet_time_head_joint(self, num_time_head):
        """Build time head for UNet-based joint models (outputs scalar)."""
        layers = []
        ch = self.hidden_channels[-1]
        conv_kwargs = {
            "in_channels": ch,
            "out_channels": ch,
            "kernel_size": 3,
            "padding": 1,
            "bias": False,
        }
        for _ in range(num_time_head - 1):
            layers += [ConvNormSilu(nn.Conv2d, num_groups=8, **conv_kwargs)]

        layers += [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, 1),
        ]
        return nn.Sequential(*layers)

    def _build_unet_data_head(self, in_channels, num_data_head, embed_dim=None):
        """Build data head for UNet-based models.

        Args:
            in_channels: Output channels (same as input image channels)
            num_data_head: Number of data head layers
            embed_dim: If provided, data head layers will condition on time embedding
        """
        ch = self.hidden_channels[0]
        return UNetDataHead(
            in_channels=ch,
            out_channels=in_channels,
            embed_dim=embed_dim,
            num_layers=num_data_head,
            num_groups=8,
        )

    def _forward_unet_time_dependent(self, t, x, cond=None, joint=True):
        """Forward pass for UNet-based time-dependent models."""
        self.nfe += 1
        if self.use_edm_precond:
            x, c_noise = self._edm_precond_input(t, x)
        else:
            c_noise = None

        # Time embedding
        if self.time_embed is not None:
            t_emb = self.time_embed(t).squeeze()
            if c_noise is not None and self.noise_embed is not None:
                t_emb = t_emb + self.noise_embed(c_noise)
            if cond is not None:
                rel_temb = self.time_embed(t - cond)
                if hasattr(self, "time_fusion"):
                    concat_emb = torch.cat([t_emb, rel_temb], dim=-1)
                    t_emb = self.time_fusion(concat_emb)
        else:
            t_emb = None

        x = self.init_conv(x, t_emb)
        skips = [x]

        # Encoder
        for block in self.encoder:
            x = block(x, t_emb)
            skips.append(x)

        # Time score
        s_t = self._compute_s_t_unet(t, x, t_emb)

        if self.joint and joint:
            # Decoder
            for i, dec_block in enumerate(self.decoder):
                x = dec_block["up_conv"](x, t_emb)
                enc_feat = skips[-(i + 2)]
                enc_feat = dec_block["adapt_conv"](enc_feat)
                x = torch.cat([x, enc_feat], dim=1)
                x = dec_block["merge_conv"](x, t_emb)
            s_x = self.out_sx(x, t_emb)
            return s_x, s_t
        return s_t

    def _build_resnet_backbone_and_heads(
        self,
        in_channels,
        hidden_channels,
        backbone,
        dropout,
        K=None,
        image_backbone_norm="batchnorm",
        nonlinearity="leakyrelu",
        num_groups=8,
        num_time_head=1,
        num_data_head=1,
        data_shape=None,
    ):
        """Build ResNet/WideResNet backbone plus time/data heads."""
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.backbone = backbone

        if self.model_type == "joint_score":
            out_dim = hidden_channels[-1]
        else:
            out_dim = 1

        if backbone == "wideresnet28_10":
            self.backbone_net = Layers.WideResNet2810(
                out_dim=out_dim,
                dropRate=dropout,
                norm_type=image_backbone_norm,
                nonlinearity=nonlinearity,
                num_groups=num_groups,
                in_channels=in_channels,
            )
        elif backbone in ["resnet18", "resnet34", "resnet50"]:
            self.backbone_net = getattr(
                Layers, f"ResNet{backbone.replace('resnet', '')}"
            )(
                out_dim=out_dim,
                norm_type=image_backbone_norm,
                nonlinearity=nonlinearity,
                num_groups=num_groups,
                in_channels=in_channels,
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        backbone_out_dim = hidden_channels[-1]

        # Time head
        if self.model_type == "joint_score":
            self.out_st = nn.Linear(backbone_out_dim, 1)

        # Data head (only for joint methods)
        if self.joint:
            if data_shape in DATA_SHAPES:
                img_channels, img_height, img_width = DATA_SHAPES[data_shape]
            else:
                img_channels, img_height, img_width = in_channels, 32, 32
            self._img_channels = img_channels
            self._img_height = img_height
            self._img_width = img_width

            if self.model_type == "joint_score":
                data_head_layers = []
                for _ in range(num_data_head - 1):
                    data_head_layers.extend(
                        [
                            nn.Linear(backbone_out_dim, backbone_out_dim),
                            nn.SiLU(),
                        ]
                    )
                data_head_layers.append(
                    nn.Linear(
                        backbone_out_dim,
                        img_channels * img_height * img_width,
                    )
                )
                self.data_head = nn.Sequential(*data_head_layers)

    def _forward_resnet_time_dependent(self, t, x, cond=None, joint=True):
        """Forward pass for ResNet-based time-dependent models."""
        self.nfe += 1
        if self.use_edm_precond:
            x, c_noise = self._edm_precond_input(t, x)
        else:
            c_noise = None

        # Time embedding
        if self.time_embed is not None:
            t_emb = self.time_embed(t).squeeze()
            if c_noise is not None and self.noise_embed is not None:
                t_emb = t_emb + self.noise_embed(c_noise)
            if cond is not None:
                rel_temb = self.time_embed(t - cond)
                if hasattr(self, "time_fusion"):
                    concat_emb = torch.cat([t_emb, rel_temb], dim=-1)
                    t_emb = self.time_fusion(concat_emb)
        else:
            t_emb = None

        # Ensure 3-channel input if backbone expects 3 channels
        if x.shape[1] == 1 and self.in_channels == 3:
            x = x.repeat(1, 3, 1, 1)

        features = self.backbone_net(x)
        s_t = self._compute_s_t_resnet(t, features, t_emb)

        if self.joint and joint and hasattr(self, "data_head"):
            s_x = self.data_head(features)
            s_x = s_x.view(
                x.shape[0], self._img_channels, self._img_height, self._img_width
            )
            return s_x, s_t
        return s_t

    # ---- s_t implementations for UNet / ResNet ----
    def _compute_s_t_unet(self, t, features, t_emb):
        raise NotImplementedError

    def _compute_s_t_resnet(self, t, features, t_emb):
        raise NotImplementedError


class ImageJointScoreModel(_BaseImageTimeDependentModel):
    """UNet-based joint score model for images (optionally other backbones if needed)."""

    def __init__(
        self,
        args,
        in_channels,
        hidden_channels,
        embed_dim=128,
        nonlinearity="leakyrelu",
        backbone="unet",
        dropout=0.0,
        path=None,
    ):
        super().__init__(args=args, path=path)
        self.model_type = "joint_score"
        self.data_type = "image"
        self.backbone = backbone
        self.time_act = getattr(args, "time_act", "leakyrelu")

        # Time embedding
        self._build_time_embedding(embed_dim)


        if backbone == "unet":
            self._build_unet_architecture(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                embed_dim=embed_dim,
                num_time_head=getattr(args, "num_time_head", 1),
            )
            # joint_score: scalar time head with configurable layers
            self.out_st = self._build_unet_time_head_joint(
                num_time_head=getattr(args, "num_time_head", 1),
            )
            if self.joint:
                self.out_sx = self._build_unet_data_head(
                    in_channels=in_channels,
                    num_data_head=getattr(args, "num_data_head", 1),
                    embed_dim=embed_dim,
                )
        else:
            # For now, we keep non-UNet joint models in the ResNet-specific class.
            raise ValueError("Use ImageJointScoreModelResNet for ResNet/WideResNet.")

        # Optional EDM noise embedding
        if self.use_edm_precond and self.edm_use_noise_embed:
            time_emb_dim = embed_dim
            self.noise_embed = nn.Sequential(
                nn.Linear(1, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )

        self._initialize_weights()

    def _compute_s_t_unet(self, t, features, t_emb):
        """Compute time score using the built time head."""
        return self.out_st(features).view(-1, 1)

    def _compute_s_t_resnet(self, t, features, t_emb):
        raise NotImplementedError("ResNet path is handled by ImageJointScoreModelResNet.")

    def forward(self, t, x, cond=None, joint=True):
        return self._forward_unet_time_dependent(t, x, cond=cond, joint=joint)


class ImageJointScoreModelResNet(_BaseImageTimeDependentModel):
    """ResNet/WideResNet-based joint score model for images."""

    def __init__(
        self,
        args,
        in_channels,
        hidden_channels,
        embed_dim=128,
        nonlinearity="leakyrelu",
        backbone="resnet18",
        dropout=0.3,
        path=None,
    ):
        super().__init__(args=args, path=path)
        self.model_type = "joint_score"
        self.data_type = "image"
        self.backbone = backbone
        self.time_act = getattr(args, "time_act", "leakyrelu")

        # Time embedding
        self._build_time_embedding(embed_dim)

        # ResNet/WideResNet backbone and heads
        self._build_resnet_backbone_and_heads(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            backbone=backbone,
            dropout=dropout,
            K=None,
            image_backbone_norm=getattr(args, "image_backbone_norm", "batchnorm"),
            nonlinearity=getattr(args, "nonlinearity", nonlinearity),
            num_groups=getattr(args, "image_backbone_num_groups", 8),
            num_time_head=getattr(args, "num_time_head", 1),
            num_data_head=getattr(args, "num_data_head", 1),
            data_shape=getattr(args, "data", None),
        )

        # Optional EDM noise embedding
        if self.use_edm_precond and self.edm_use_noise_embed:
            time_emb_dim = embed_dim
            self.noise_embed = nn.Sequential(
                nn.Linear(1, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )

        self._initialize_weights()

    def _compute_s_t_unet(self, t, features, t_emb):
        raise NotImplementedError("UNet path is handled by ImageJointScoreModel.")

    def _compute_s_t_resnet(self, t, features, t_emb):
        """Same as UnifiedModel._compute_s_t_resnet_joint."""
        return self.out_st(features).view(-1, 1)

    def forward(self, t, x, cond=None, joint=True):
        return self._forward_resnet_time_dependent(t, x, cond=cond, joint=joint)


class ResNetDensityRatioModel(_BaseImageModel):
    """ResNet/WideResNet-based density-ratio model for images."""

    def __init__(self, in_channels=3, num_classes=1, backbone="resnet18", dropout=0.3):
        # For density ratio we do not need joint training, so args can be a minimal dummy.
        class DummyArgs:
            joint = False

        super().__init__(args=DummyArgs(), path=None)
        self.model_type = "density_ratio"
        self.data_type = "image"
        self.backbone = backbone
        self.in_channels = in_channels

        # Build backbone identical to UnifiedModel._build_resnet_architecture for density_ratio
        norm_type = "batchnorm"
        nonlinearity = "leakyrelu"
        num_groups = 8
        hidden_channels = [64, 128, 256, 512]

        if backbone == "wideresnet28_10":
            self.backbone_net = Layers.WideResNet2810(
                out_dim=1,
                dropRate=dropout,
                norm_type=norm_type,
                nonlinearity=nonlinearity,
                num_groups=num_groups,
                in_channels=in_channels,
            )
        elif backbone in ["resnet18", "resnet34", "resnet50"]:
            self.backbone_net = getattr(
                Layers, f"ResNet{backbone.replace('resnet', '')}"
            )(
                out_dim=1,
                norm_type=norm_type,
                nonlinearity=nonlinearity,
                num_groups=num_groups,
                in_channels=in_channels,
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.hidden_channels = hidden_channels
        self._initialize_weights()

    def _forward_density_ratio(self, x):
        """Same as UnifiedModel._forward_density_ratio_image_resnet."""
        if x.shape[1] == 1 and self.in_channels == 3:
            x = x.repeat(1, 3, 1, 1)
        logits = self.backbone_net(x)
        if logits.size(-1) == 1:
            return logits.squeeze(-1)
        return logits

    def forward(self, x, cond=None, joint=True):
        return self._forward_density_ratio(x)
