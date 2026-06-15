# -*- coding: utf-8 -*-
"""F-divergence: change point detection."""
import torch
from torch.utils.data import Dataset, DataLoader

from dre_utils import data_utils
from .base import DataModule


CHANGEPOINT_DATASETS = ['cpd_mc', 'cpd_text', 'cpd_creditcard']


class DREPairDataset(Dataset):
    def __init__(self, generator, num_samples: int, batch_size: int):
        self.generator = generator
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_batches = num_samples // batch_size

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        px, qx = self.generator.sample(batch_size=self.batch_size)
        return px, qx, torch.tensor(0.0)


class FullSequenceDataset(Dataset):
    def __init__(self, generator, data_type: str):
        if data_type == "cpd_mc":
            self.data = generator.sample_monte_carlo(num_samples=2000, d=generator.d, experiment_id=999)
        elif data_type == "cpd_text":
            self.data = generator.load_and_process_text_data()
        elif data_type == "cpd_creditcard":
            self.data = generator.load_and_process_creditcard_data()
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.data


class ChangePointDetectionDataModule(DataModule):
    def __init__(self, args):
        super().__init__(args)
        self.num_workers = 0
        self.batch_size = args.batch_size
        self.num_samples = getattr(args, 'num_samples', 10000)
        self.d = getattr(args, 'd', 6)
        self.seed = args.seed
        self.generator = data_utils.ChangePointDetectionDataGenerator(
            data_type=self.data_type, d=self.d, rng=None, args=args
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = DREPairDataset(
                generator=self.generator, num_samples=self.num_samples, batch_size=self.batch_size
            )
        if stage == "validate" or stage is None:
            self.val_dataset = FullSequenceDataset(generator=self.generator, data_type=self.data_type)
        if stage == "test" or stage is None:
            self.test_dataset = FullSequenceDataset(generator=self.generator, data_type=self.data_type)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, num_workers=0, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=0, pin_memory=True)
