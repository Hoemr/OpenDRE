# -*- coding: utf-8 -*-
"""F-divergence: mutual information (synthetic)."""
from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader

from utils import data_utils
from .base import DataModule, IterDataModule


MI_DATASETS = ['half_cube_map', 'asinh_mapping', 'additive_noise', 'gamma_exponential', 'diag_multi_gauss', 'gauss']


class MutualInformationDataset(Dataset):
    def __init__(self, data_generator, rho, batch_size=5000, batch_num=1, d=2, data_type="gauss", rng=None):
        joint_data, marginal_data, true_mi = data_generator(
            rho=rho, batch_size=batch_size, batch_num=batch_num, d=d, data_type=data_type, rng=rng
        )
        self.joint_data = joint_data
        self.marginal_data = marginal_data
        self.true_mi = true_mi

    def __len__(self):
        return self.joint_data.size(0)

    def __getitem__(self, idx):
        return self.joint_data[idx], self.marginal_data[idx], self.true_mi


class MutualInformationDataModule(DataModule):
    def __init__(self, args):
        super().__init__(args)
        self.batch_num = args.batch_num
        self.d = args.d
        self.rho = args.rho
        self.num_workers = 4

    def setup(self, stage=None):
        rng = np.random.RandomState(seed=self.seed)
        self.train_dataset = MutualInformationDataset(
            data_utils.generate_MI_data, rho=self.rho,
            batch_size=self.batch_size, batch_num=self.batch_num,
            d=self.d, data_type=self.data_type, rng=rng,
        )
        self.val_dataset = MutualInformationDataset(
            data_utils.generate_MI_data, rho=self.rho,
            batch_size=self.test_batch_size, batch_num=1,
            d=self.d, data_type=self.data_type, rng=rng,
        )
        self.test_dataset = self.val_dataset


class MutualInformationIterDataset(IterableDataset):
    def __init__(self, data_generator, rho, batch_size=5000, batch_num=1, d=2, data_type="gauss", rng=None):
        self.batch_num = batch_num
        self.generator = partial(
            data_generator, rho=rho, batch_size=batch_size, batch_num=10, d=d, data_type=data_type, rng=rng
        )

    def __iter__(self):
        for _ in range(self.batch_num):
            joint_data, marginal_data, true_mi = self.generator()
            yield joint_data.type(torch.float32), marginal_data.type(torch.float32), true_mi


class MutualInformationIterDataModule(IterDataModule):
    def __init__(self, args):
        super().__init__(args)
        self.rho = args.rho

    def setup(self, stage=None):
        rng = np.random.RandomState(seed=self.seed)
        self.train_dataset = MutualInformationIterDataset(
            data_utils.generate_MI_data, rho=self.rho,
            batch_size=self.batch_size, batch_num=self.batch_num,
            d=self.d, data_type=self.data_type, rng=rng,
        )
        self.val_dataset = MutualInformationIterDataset(
            data_utils.generate_MI_data, rho=self.rho,
            batch_size=self.test_batch_size, batch_num=1,
            d=self.d, data_type=self.data_type, rng=rng,
        )
        self.test_dataset = self.val_dataset
