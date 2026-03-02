# -*- coding: utf-8 -*-
"""Base DataModules: DataModule (map-style) and IterDataModule (iterable)."""
from abc import ABC, abstractmethod

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.seed = args.seed
        self.data_type = args.data
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.num_workers = 5

    def prepare_data(self):
        pass

    @abstractmethod
    def setup(self, stage=None):
        pass

    def _create_dataloader(self, dataset, batch_size=64, shuffle=False):
        num_workers = self.num_workers
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False, persistent_workers=num_workers > 1)

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset, batch_size=self.test_batch_size)

    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False)


class IterDataModule(pl.LightningDataModule, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.seed = args.seed
        self.data_type = args.data
        self.batch_size = args.batch_size
        self.batch_num = args.batch_num
        self.test_batch_size = args.test_batch_size
        self.d = args.d
        self.num_workers = 0

    def prepare_data(self):
        pass

    @abstractmethod
    def setup(self, stage=None):
        pass

    def _create_dataloader(self, dataset):
        num_workers = self.num_workers
        return DataLoader(dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True, persistent_workers=num_workers > 1)

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset)
