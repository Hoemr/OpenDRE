# -*- coding: utf-8 -*-
"""Two sample test."""
from functools import partial

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .base import IterDataModule

def generate_two_sample_test_data(batch_size, d, mu0=0, sigma0=1, mu1=0, sigma1=1, data_type="gauss_two_sample", rng=None):
    if rng is None:
        rng = np.random.RandomState()
    mean0 = np.ones(d) * mu0
    mean1 = np.ones(d) * mu1
    cov0 = np.eye(d) * sigma0
    cov1 = np.eye(d) * sigma1
    if data_type == "gauss_two_sample":
        qx = rng.multivariate_normal(mean1, cov1, batch_size)
        px = rng.multivariate_normal(mean0, cov0, batch_size)
    else:
        raise ValueError("Unsupported distribution: %s" % data_type)
    return torch.from_numpy(qx).type(torch.float32), torch.from_numpy(px).type(torch.float32), 0

class TwoSampleTestIterDataset(IterableDataset):
    def __init__(self, data_generator, batch_size=5000, batch_num=1, d=2, mu0=0, sigma0=1, mu1=0, sigma1=1, data_type="gauss_two_sample", rng=None):
        self.batch_num = batch_num
        self.generator = partial(data_generator, batch_size=batch_size, d=d, mu0=mu0, sigma0=sigma0, mu1=mu1, sigma1=sigma1, data_type=data_type, rng=rng)

    def __iter__(self):
        for _ in range(self.batch_num):
            qx, px, _ = self.generator()
            yield qx.type(torch.float32), px.type(torch.float32), 0

class TwoSampleTestIterDataModule(IterDataModule):
    def __init__(self, args):
        super().__init__(args)
        self.mu0 = args.mu0
        self.sigma0 = args.sigma0
        self.mu1 = args.mu1
        self.sigma1 = args.sigma1

    def setup(self, stage=None):
        rng = np.random.RandomState(seed=self.seed)
        self.train_dataset = TwoSampleTestIterDataset(
            generate_two_sample_test_data, batch_size=self.batch_size, batch_num=self.batch_num,
            d=self.d, mu0=self.mu0, sigma0=self.sigma0, mu1=self.mu1, sigma1=self.sigma1,
            data_type=self.data_type, rng=rng)
        self.val_dataset = TwoSampleTestIterDataset(
            generate_two_sample_test_data, batch_size=self.test_batch_size, batch_num=1,
            d=self.d, mu0=self.mu0, sigma0=self.sigma0, mu1=self.mu1, sigma1=self.sigma1,
            data_type=self.data_type, rng=rng)
        self.test_dataset = self.val_dataset
