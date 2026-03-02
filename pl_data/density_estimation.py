# -*- coding: utf-8 -*-
"""Likelihood estimation: tabular and synthetic2d."""
from functools import partial

import numpy as np
from torch.utils.data import Dataset, IterableDataset, DataLoader

from core import data_utils
from .base import DataModule, IterDataModule


TABULAR_DATASETS = ['bsds300', 'power', 'gas', 'hepmass', 'miniboone']
SYNTHETIC2D_DATASETS = ['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings', 'tree']


class TabularDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TabularDataModule(DataModule):
    def __init__(self, args):
        super().__init__(args)
        self.d = args.d
        self.generator = data_utils.LikelihoodDataGenerator(self.data_type, args=args)

    def setup(self, stage=None):
        if (stage == "fit" or stage == "validate") or stage is None:
            self.train_dataset = TabularDataset(self.generator.dataset.trn.x)
            self.val_dataset = TabularDataset(self.generator.dataset.val.x)
        if stage == "test" or stage is None:
            self.test_dataset = TabularDataset(self.generator.dataset.tst.x)


class Synthetic2DDataset(IterableDataset):
    def __init__(self, data_generator, batch_size=5000, batch_num=1):
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.generator = partial(data_generator, num_samples=batch_size)

    def __iter__(self):
        for _ in range(self.batch_num):
            yield self.generator()


class Synthetic2DIterDataModule(IterDataModule):
    def __init__(self, args):
        super().__init__(args)

    def setup(self, stage=None):
        rng = np.random.RandomState(seed=self.seed)
        self.generator = data_utils.LikelihoodDataGenerator(
            data_type=self.data_type, d=2, rng=rng, args=self.args
        )
        self.train_dataset = Synthetic2DDataset(
            self.generator.sample,
            batch_size=self.batch_size,
            batch_num=self.batch_num,
        )
        self.val_dataset = Synthetic2DDataset(
            self.generator.sample,
            batch_size=self.test_batch_size,
            batch_num=1,
        )
        self.test_dataset = self.val_dataset
