# -*- coding: utf-8 -*-
"""MI real (image/text/mixture)."""
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from core import data_utils
from .base import DataModule

MI_REAL_DATASETS = ['image', 'text', 'mixture']


class MIRealDataset(Dataset):
    def __init__(self, generator, target_mi, num_samples=10000, batch_size=64):
        self.generator = generator
        self.target_mi = target_mi
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.bsc_p = generator.cal_bsc(generator.ds, target_mi)
        self._generate_data()

    def _generate_data(self):
        batch_num = (self.num_samples + self.batch_size - 1) // self.batch_size
        self.data_pairs = []
        for _ in range(batch_num):
            x_batch, y_batch = self.generator.generate_batch(self.batch_size, bsc_p=self.bsc_p)
            self.data_pairs.extend(list(zip(x_batch, y_batch)))
        self.data_pairs = self.data_pairs[:self.num_samples]

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        x, y = self.data_pairs[idx]
        return x, y, torch.tensor(self.target_mi, dtype=torch.float32)


class MIRealIterDataset(IterableDataset):
    def __init__(self, generator, target_mi, batch_size=64, batch_num=1000):
        self.generator = generator
        self.target_mi = target_mi
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.bsc_p = generator.cal_bsc(generator.ds, target_mi)

    def __iter__(self):
        for _ in range(self.batch_num):
            x_batch, y_batch = self.generator.generate_batch(self.batch_size, bsc_p=self.bsc_p)
            for i in range(len(x_batch)):
                yield x_batch[i], y_batch[i], torch.tensor(self.target_mi, dtype=torch.float32)


class StepwiseMIIterDataset(IterableDataset):
    def __init__(self, generator, mi_values, steps_per_mi, batch_size):
        self.generator = generator
        self.mi_values = mi_values
        self.steps_per_mi = steps_per_mi
        self.batch_size = batch_size
        self.bsc_params = [generator.cal_bsc(generator.ds, mi) for mi in mi_values]

    def __iter__(self):
        for mi_idx, target_mi in enumerate(self.mi_values):
            bsc_p = self.bsc_params[mi_idx]
            for step in range(self.steps_per_mi):
                x_batch, y_batch = self.generator.generate_batch(self.batch_size, bsc_p=bsc_p)
                for i in range(len(x_batch)):
                    yield x_batch[i], y_batch[i], torch.tensor(target_mi, dtype=torch.float32)


class MIRealDataModule(DataModule):
    def __init__(self, args):
        super().__init__(args)
        self.num_workers = 0
        self.generator = data_utils.MIRealDatasetGenerator(
            data=args.data, dname1=args.mi_dname1, dname2=args.mi_dname2,
            ds=args.mi_ds, dr=args.mi_dr, image_channels=args.mi_image_channels,
            image_patches=eval(args.mi_image_patches), nuisance=args.mi_nuisance, data_dir=args.data_dir)

    def setup(self, stage=None):
        if (stage == "fit" or stage == "validate") or stage is None:
            if self.args.mi_mode == "stepwise":
                self.train_dataset = self._create_stepwise_dataset()
            else:
                self.train_dataset = MIRealDataset(
                    self.generator, target_mi=self.args.true_mi,
                    num_samples=self.args.mi_n_samples, batch_size=self.batch_size)
            self.val_dataset = MIRealDataset(
                self.generator, target_mi=self.args.true_mi, num_samples=5000, batch_size=self.test_batch_size)
        if stage == "test" or stage is None:
            self.test_dataset = self.val_dataset

    def _create_stepwise_dataset(self):
        mi_values = [2, 4, 6, 8, 10]
        steps_per_mi = self.args.mi_n_steps // len(mi_values)
        return StepwiseMIIterDataset(self.generator, mi_values, steps_per_mi, self.batch_size)

    def _create_dataloader(self, dataset, batch_size=64, shuffle=False):
        num_workers = self.num_workers
        if isinstance(dataset, IterableDataset):
            return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 1)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 1)
