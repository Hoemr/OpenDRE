# -*- coding: utf-8 -*-
"""Two sample test: DRE model."""
import torch
from .base import DensityRatioEstimationModel

class TwoSampleTestModel(DensityRatioEstimationModel):
    def __init__(self, args, save_path, datamodule):
        super().__init__(args, save_path, datamodule)

    def calculate_pearson_divergence(self, sample1, sample2):
        n_s = len(sample1)
        n_t = len(sample2)
        density_ratio_estimator = lambda x: torch.exp(self.density_ratio_fn(self.model, x, joint=None)[0].squeeze())
        r_s = density_ratio_estimator(sample1)
        r_t = density_ratio_estimator(sample2)
        return (0.5 / n_s) * torch.sum(r_s) - (1 / n_t) * torch.sum(r_t) + 0.5

    def permutation_test(self, sample1, sample2, K=100):
        original_divergence = self.calculate_pearson_divergence(sample1, sample2)
        combined = torch.cat([sample1, sample2])
        n_s = len(sample1)
        n_t = len(sample2)
        divergences = []
        for _ in range(K):
            permuted = combined[torch.randperm(len(combined))]
            new_sample1 = permuted[:n_s]
            new_sample2 = permuted[n_s:]
            divergence = self.calculate_pearson_divergence(new_sample1, new_sample2)
            divergences.append(divergence)
        divergences = torch.tensor(divergences)
        p_value = torch.sum(divergences >= original_divergence) / K
        return p_value
