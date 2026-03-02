# -*- coding: utf-8 -*-
"""F-divergence: distribution shift."""
import numpy as np
from torch.utils.data import IterableDataset

from core import data_utils
from .base import IterDataModule

DISTRIBUTION_SHIFT_DATASETS = ['gauss_shift', 'beta_shift', 'cifar_c', 'domainbed', 'gauss_shift_kl']


class DistributionShiftIterDataset(IterableDataset):
    def __init__(self, generator, batch_size=5000, batch_num=1, args=None, **kwargs):
        self.generator = generator
        self.batch_size = batch_size
        self.batch_num = batch_num

    def __iter__(self):
        for _ in range(self.batch_num):
            yield self.generator.sample(batch_size=self.batch_size)


class DistributionShiftIterDataModule(IterDataModule):
    def __init__(self, args):
        super().__init__(args)
        self.shift_step = 0
        self.shift = args.shift
        self.generator = None

    def setup(self, stage=None):
        rng = np.random.RandomState(seed=self.seed)
        self.generator = data_utils.DistributionShiftDatasetGenerator(
            data_type=self.data_type, d=self.d, rng=rng, args=self.args
        )
        self.train_dataset = DistributionShiftIterDataset(
            self.generator, batch_size=self.batch_size, batch_num=self.batch_num, args=self.args
        )
        self.val_dataset = DistributionShiftIterDataset(
            self.generator, batch_size=self.test_batch_size, batch_num=1, args=self.args
        )
        self.test_dataset = self.val_dataset
