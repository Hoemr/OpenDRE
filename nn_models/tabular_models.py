# -*- coding: utf-8 -*-
"""Tabular-data neural network models: joint score and density ratio."""
from .unified import UnifiedModel


class JointScoreModel(UnifiedModel):
    def __init__(self, input_dim, hidden_dims, embed_dim=256, nonlinearity="leakyrelu", args=None, norm=False, use_spectral_norm=False, dynamic=False):
        super().__init__(model_type='joint_score', data_type='tabular', backbone='mlp', input_dim=input_dim, hidden_dims=hidden_dims, embed_dim=embed_dim, nonlinearity=nonlinearity, args=args, norm=norm, use_spectral_norm=use_spectral_norm, dynamic=dynamic)


class DecompJointScoreModel(UnifiedModel):
    """Tabular decomp joint score (polynomial decomposition)."""
    def __init__(self, input_dim, hidden_dims, embed_dim=256, K=300, polynomials="gaussrbf", noise_type="zero", nonlinearity="leakyrelu", args=None, norm=False, use_spectral_norm=False, dynamic=False):
        super().__init__(model_type='decomp_joint_score', data_type='tabular', backbone='mlp', input_dim=input_dim, hidden_dims=hidden_dims, embed_dim=embed_dim, K=K, polynomials=polynomials, noise_type=noise_type, nonlinearity=nonlinearity, args=args, norm=norm, use_spectral_norm=use_spectral_norm, dynamic=dynamic)


class NeuralDensityRatioModel(UnifiedModel):
    def __init__(self, input_dim, hidden_dims, nonlinearity="leakyrelu", args=None, norm=False, use_spectral_norm=False, dynamic=False):
        super().__init__(model_type='density_ratio', data_type='tabular', backbone='mlp', input_dim=input_dim, hidden_dims=hidden_dims, nonlinearity=nonlinearity, args=args, norm=norm, use_spectral_norm=use_spectral_norm, dynamic=dynamic)
