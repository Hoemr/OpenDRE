# -*- coding: utf-8 -*-
"""Thin compatibility wrapper for legacy UnifiedModel.

Historically, OS_DRE used a single ``UnifiedModel`` class to handle all
combinations of:

- data type: tabular vs image
- model type: joint_score, decomp_joint_score, density_ratio
- image backbone: UNet, ResNet-18/34/50, WideResNet-28-10

This led to a very large and hard-to-maintain module. The implementation has
now been refactored into task-specific modules:

- ``nn_models/tabular_models.py`` for tabular score/density-ratio models
- ``nn_models/image_models.py`` for image-based UNet/ResNet/WideResNet models

The class below exists only to preserve the old import path
``from nn_models import UnifiedModel`` for legacy scripts. New code should
import the concrete models directly from ``tabular_models`` or ``image_models``.
"""

from typing import Optional, Sequence

import torch.nn as nn

from .tabular_models import (
    JointScoreModel,
    DecompJointScoreModel,
    NeuralDensityRatioModel,
)
from .image_models import (
    ImageJointScoreModel,
    ImageDecompJointScoreModel,
    ImageJointScoreModelResNet,
    ImageDecompJointScoreModelResNet,
    ResNetDensityRatioModel,
)


class UnifiedModel(nn.Module):
    """Compatibility wrapper that dispatches to concrete tabular/image models.

    Args mirror the historical UnifiedModel signature, but the implementation
    simply builds and delegates to the corresponding model class:

    - tabular + joint_score          -> JointScoreModel
    - tabular + decomp_joint_score   -> DecompJointScoreModel
    - tabular + density_ratio        -> NeuralDensityRatioModel
    - image + joint_score, backbone='unet'
        -> ImageJointScoreModel
    - image + decomp_joint_score, backbone='unet'
        -> ImageDecompJointScoreModel
    - image + joint_score, backbone in {resnet18, resnet34, resnet50, wideresnet28_10}
        -> ImageJointScoreModelResNet
    - image + decomp_joint_score, backbone in {resnet18, resnet34, resnet50, wideresnet28_10}
        -> ImageDecompJointScoreModelResNet
    - image + density_ratio, any supported ResNet/WideResNet backbone
        -> ResNetDensityRatioModel
    """

    def __init__(
        self,
        model_type: str = "joint_score",
        data_type: str = "tabular",
        backbone: str = "mlp",
        input_dim: Optional[int] = None,
        in_channels: Optional[int] = None,
        hidden_dims: Optional[Sequence[int]] = None,
        hidden_channels: Optional[Sequence[int]] = None,
        K: int = 300,
        embed_dim: int = 256,
        nonlinearity: str = "leakyrelu",
        polynomials: str = "imqrbf",
        noise_type: str = "zero",
        args=None,
        norm: bool = False,
        use_spectral_norm: bool = False,
        dynamic: bool = False,
        dropout: float = 0.3,
        path=None,
    ):
        super().__init__()

        # Store a reference to the concrete underlying model.
        self.model_type = model_type
        self.data_type = data_type
        self.backbone = backbone

        if data_type == "tabular":
            if input_dim is None or hidden_dims is None:
                raise ValueError(
                    "For tabular models, `input_dim` and `hidden_dims` must be provided."
                )
            if model_type == "joint_score":
                self.model = JointScoreModel(
                    input_dim=input_dim,
                    hidden_dims=list(hidden_dims),
                    embed_dim=embed_dim,
                    nonlinearity=nonlinearity,
                    args=args,
                    norm=norm,
                    use_spectral_norm=use_spectral_norm,
                    dynamic=dynamic,
                    path=path,
                )
            elif model_type == "decomp_joint_score":
                self.model = DecompJointScoreModel(
                    input_dim=input_dim,
                    hidden_dims=list(hidden_dims),
                    embed_dim=embed_dim,
                    K=K,
                    polynomials=polynomials,
                    noise_type=noise_type,
                    nonlinearity=nonlinearity,
                    args=args,
                    norm=norm,
                    use_spectral_norm=use_spectral_norm,
                    dynamic=dynamic,
                    path=path,
                )
            elif model_type == "density_ratio":
                self.model = NeuralDensityRatioModel(
                    input_dim=input_dim,
                    hidden_dims=list(hidden_dims),
                    nonlinearity=nonlinearity,
                    args=args,
                    norm=norm,
                    use_spectral_norm=use_spectral_norm,
                    dynamic=dynamic,
                )
            else:
                raise ValueError(f"Unsupported model_type for tabular data: {model_type}")

        elif data_type == "image":
            if in_channels is None or (hidden_channels is None and model_type != "density_ratio"):
                raise ValueError(
                    "For image models, `in_channels` (and `hidden_channels` for score models) must be provided."
                )

            if model_type in ["joint_score", "decomp_joint_score"]:
                if backbone == "unet":
                    if model_type == "joint_score":
                        self.model = ImageJointScoreModel(
                            args=args,
                            in_channels=in_channels,
                            hidden_channels=list(hidden_channels),
                            embed_dim=embed_dim,
                            nonlinearity=nonlinearity,
                            backbone="unet",
                            dropout=dropout,
                            path=path,
                        )
                    else:  # decomp_joint_score
                        self.model = ImageDecompJointScoreModel(
                            args=args,
                            in_channels=in_channels,
                            hidden_channels=list(hidden_channels),
                            embed_dim=embed_dim,
                            K=K,
                            nonlinearity=nonlinearity,
                            polynomials=polynomials,
                            noise_type=noise_type,
                            path=path,
                        )
                elif backbone in [
                    "resnet18",
                    "resnet34",
                    "resnet50",
                    "wideresnet28_10",
                ]:
                    if model_type == "joint_score":
                        self.model = ImageJointScoreModelResNet(
                            args=args,
                            in_channels=in_channels,
                            hidden_channels=list(hidden_channels),
                            embed_dim=embed_dim,
                            nonlinearity=nonlinearity,
                            backbone=backbone,
                            dropout=dropout,
                            path=path,
                        )
                    else:  # decomp_joint_score
                        self.model = ImageDecompJointScoreModelResNet(
                            args=args,
                            in_channels=in_channels,
                            hidden_channels=list(hidden_channels),
                            embed_dim=embed_dim,
                            K=K,
                            nonlinearity=nonlinearity,
                            polynomials=polynomials,
                            noise_type=noise_type,
                            backbone=backbone,
                            dropout=dropout,
                            path=path,
                        )
                else:
                    raise ValueError(f"Unsupported image backbone: {backbone}")

            elif model_type == "density_ratio":
                # For image density-ratio models we always use a ResNet/WideResNet-style backbone.
                self.model = ResNetDensityRatioModel(
                    in_channels=in_channels,
                    num_classes=1,
                    backbone=backbone,
                    dropout=dropout,
                )
            else:
                raise ValueError(f"Unsupported model_type for image data: {model_type}")

        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

        # Decomp-only: attach methods and Pk so they exist only on decomp instances.
        # Non-decomp models have no compute_fk / time_gradient / get_fk_reg / set_fk / Pk.
        if self.model_type == "decomp_joint_score":
            self.compute_fk = self.model.compute_fk
            self.compute_weighted_sum = self.model.compute_weighted_sum
            self.time_gradient = self.model.time_gradient
            self.get_fk_reg = self.model.get_fk_reg
            self.set_fk = self.model.set_fk
            self.Pk = self.model.Pk

    # ---- Core delegation ----
    def forward(self, *inputs, **kwargs):
        """Delegate forward pass to the underlying concrete model."""
        return self.model(*inputs, **kwargs)

    def reset_states(self):
        if hasattr(self.model, "reset_states"):
            return self.model.reset_states()

