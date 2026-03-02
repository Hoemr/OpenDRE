# -*- coding: utf-8 -*-
"""Image-data neural network models: UNet/ResNet joint score and density ratio."""
from .unified import UnifiedModel


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
        class DummyArgs:
            joint = False
        super().__init__(
            model_type='density_ratio',
            data_type='image',
            backbone=backbone,
            in_channels=in_channels,
            hidden_channels=[64, 128, 256, 512],
            embed_dim=128,
            args=DummyArgs(),
            dropout=dropout
        )
