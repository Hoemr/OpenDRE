# -*- coding: utf-8 -*-
"""OOD detection datasets and DataModule."""
import os
import random

import numpy as np
import torch
import torchvision.datasets as dset
from torch.utils.data import Dataset
from torchvision import transforms

from .base import DataModule

OOD_IN_DIST_DATASETS = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']
OOD_BASE_TYPES = ['local', 'general', 'universal']

OOD_EVAL_CONFIG = {
    'mnist': {'ood_datasets': ['fashion_mnist', 'emnist']},
    'fashion_mnist': {'ood_datasets': ['mnist', 'emnist']},
    'cifar10': {'ood_datasets': ['svhn', 'cifar100', 'lsun', 'celeba']},
    'cifar100': {'ood_datasets': ['svhn', 'cifar10', 'lsun', 'celeba']},
}


class ConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.cumulative_sizes = []
        total_size = 0
        for dataset in datasets:
            total_size += len(dataset)
            self.cumulative_sizes.append(total_size)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = 0
        for i, cumulative_size in enumerate(self.cumulative_sizes):
            if idx < cumulative_size:
                dataset_idx = i
                break
        if dataset_idx == 0:
            actual_idx = idx
        else:
            actual_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][actual_idx]


class OODDetectionDataset(Dataset):
    def __init__(self, in_dist_dataset, ood_dataset, noise_std=0.0):
        self.in_dist_dataset = in_dist_dataset
        self.ood_dataset = ood_dataset
        self.noise_std = noise_std
        self.num_samples = len(in_dist_dataset)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x_in, _ = self.in_dist_dataset[idx]
        ood_idx = random.randint(0, len(self.ood_dataset) - 1)
        x_ood, _ = self.ood_dataset[ood_idx]
        if self.noise_std > 0:
            x_in = x_in + torch.randn_like(x_in) * self.noise_std
            x_ood = x_ood + torch.randn_like(x_ood) * self.noise_std
        return x_in, x_ood, torch.tensor(0.0)


class OutOfDistributionDetectionDataModule(DataModule):
    def __init__(self, args):
        super().__init__(args)
        self.ood_in_dist = getattr(args, 'ood_in_dist', 'mnist')
        self.ood_base_type = getattr(args, 'ood_base_type', 'universal')
        self.ood_noise_std = getattr(args, 'ood_noise_std', 0.1)
        self.ood_anneal_alpha_init = getattr(args, 'ood_anneal_alpha_init', 0.0)
        self.ood_crop_size = getattr(args, 'ood_crop_size', 12)
        self.num_workers = 0

    def prepare_data(self):
        if self.ood_in_dist == 'mnist':
            dset.MNIST(self.args.data_dir, train=True, download=True)
            dset.MNIST(self.args.data_dir, train=False, download=True)
        elif self.ood_in_dist == 'fashion_mnist':
            dset.FashionMNIST(self.args.data_dir, train=True, download=True)
            dset.FashionMNIST(self.args.data_dir, train=False, download=True)
        elif self.ood_in_dist == 'cifar10':
            dset.CIFAR10(self.args.data_dir, train=True, download=True)
            dset.CIFAR10(self.args.data_dir, train=False, download=True)
        elif self.ood_in_dist == 'cifar100':
            dset.CIFAR100(self.args.data_dir, train=True, download=True)
            dset.CIFAR100(self.args.data_dir, train=False, download=True)

    def setup(self, stage=None):
        transform_train = self._get_transform(train=True)
        transform_test = self._get_transform(train=False)
        if (stage == "fit" or stage == "validate") or stage is None:
            in_dist_dataset = self._get_in_dist_dataset(train=True, transform=transform_train)
            ood_dataset = self._get_ood_dataset(train=True, transform=transform_train)
            self.train_dataset = OODDetectionDataset(in_dist_dataset, ood_dataset, noise_std=self.ood_noise_std)
            self.val_dataset = DummyDataset(input_shape=(3, 32, 32))
        if stage == "test" or stage is None:
            self.test_dataset = DummyDataset(input_shape=(3, 32, 32))

    def _get_transform(self, train=True):
        if self.ood_in_dist in ['mnist', 'fashion_mnist']:
            transform_list = [
                transforms.Resize(32),
                transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            ]
        else:
            transform_list = [
                transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
            ]
        return transforms.Compose(transform_list)

    def _get_crop_transform(self):
        from torchvision.transforms.functional import InterpolationMode
        if self.ood_in_dist in ['mnist', 'fashion_mnist']:
            return transforms.Compose([
                transforms.Resize(32),
                transforms.RandomHorizontalFlip(),
                transforms.Pad(2),
                transforms.RandomCrop(32 - self.ood_crop_size),
                transforms.Resize(32, interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            ])
        else:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32 - self.ood_crop_size),
                transforms.Resize(32, interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ])

    def _get_in_dist_dataset(self, train=True, transform=None):
        if transform is None:
            transform = self._get_transform(train=train)
        if self.ood_in_dist == 'mnist':
            return dset.MNIST(self.args.data_dir, train=train, download=False, transform=transform)
        elif self.ood_in_dist == 'fashion_mnist':
            return dset.FashionMNIST(self.args.data_dir, train=train, download=False, transform=transform)
        elif self.ood_in_dist == 'cifar10':
            return dset.CIFAR10(self.args.data_dir, train=train, download=False, transform=transform)
        elif self.ood_in_dist == 'cifar100':
            return dset.CIFAR100(self.args.data_dir, train=train, download=False, transform=transform)

    def _get_ood_dataset(self, train=True, transform=None):
        if self.ood_base_type == 'local':
            crop_transform = self._get_crop_transform()
            return self._get_in_dist_dataset(train=train, transform=crop_transform)
        elif self.ood_base_type == 'general':
            ood_list = {'mnist': 'fashion_mnist', 'fashion_mnist': 'mnist', 'cifar10': 'svhn', 'cifar100': 'svhn'}
            ood_name = ood_list.get(self.ood_in_dist, 'svhn')
            return self._get_ood_by_name(ood_name, train=train, transform=transform)
        elif self.ood_base_type == 'universal':
            oe_file = os.path.join(self.args.data_dir, "300K_random_images.npy")
            return OODRandomImagesDataset(oe_file, transform=transform)

    def _get_ood_by_name(self, ood_name, train=True, transform=None):
        if transform is None:
            transform = self._get_transform(train)
        if ood_name == 'mnist':
            return dset.MNIST(self.args.data_dir, train=train, download=False, transform=transform)
        elif ood_name == 'fashion_mnist':
            return dset.FashionMNIST(self.args.data_dir, train=train, download=False, transform=transform)
        elif ood_name == 'cifar10':
            return dset.CIFAR10(self.args.data_dir, train=train, download=False, transform=transform)
        elif ood_name == 'cifar100':
            return dset.CIFAR100(self.args.data_dir, train=train, download=False, transform=transform)
        elif ood_name == 'svhn':
            return dset.SVHN(self.args.data_dir, split='train' if train else 'test', download=False, transform=transform)
        elif ood_name == 'lsun':
            lsun_path = os.path.join(self.args.data_dir, 'LSUN_resize')
            return dset.LSUN(lsun_path, transform=transform)
        elif ood_name == 'celeba':
            return dset.CelebA(root=self.args.data_dir, split="test" if not train else "train", transform=transform, download=False)

    def _load_ood_benchmarks(self, ood_names, train=False, transform=None):
        loaded_datasets = []
        for ood_name in ood_names:
            try:
                ood_dataset = self._get_ood_by_name(ood_name, train=train, transform=transform)
                if ood_dataset is not None:
                    loaded_datasets.append(ood_dataset)
            except Exception as e:
                print("Warning: Could not load OOD dataset '%s': %s" % (ood_name, e))
                continue
        return loaded_datasets

    def _get_eval_ood_dataset(self, train=False, transform=None):
        eval_ood_datasets = OOD_EVAL_CONFIG.get(self.ood_in_dist, {}).get('ood_datasets', [])
        loaded_datasets = self._load_ood_benchmarks(eval_ood_datasets, train=train, transform=transform)
        if not loaded_datasets:
            print("Warning: No evaluation OOD datasets could be loaded. Falling back to original OOD dataset.")
            return self._get_ood_dataset(train=train, transform=transform)
        if len(loaded_datasets) == 1:
            return loaded_datasets[0]
        return ConcatDataset(loaded_datasets)

    def get_eval_ood_datasets(self, stage='val'):
        eval_ood_datasets = OOD_EVAL_CONFIG.get(self.ood_in_dist, {}).get('ood_datasets', [])
        transform_test = self._get_transform(train=False)
        return self._load_ood_benchmarks(eval_ood_datasets, train=False, transform=transform_test)


class DummyDataset(Dataset):
    def __init__(self, input_shape=(3, 32, 32)):
        self.input_shape = input_shape

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torch.zeros(self.input_shape), 0


class OODRandomImagesDataset(Dataset):
    def __init__(self, npy_file, transform=None):
        self.data = np.load(npy_file)
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img = self.data[idx]
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            img = torch.from_numpy(img).float() / 255.0
            if img.dim() == 2:
                img = img.unsqueeze(0).repeat(3, 1, 1)
            elif img.dim() == 3 and img.shape[0] != 3:
                img = img.permute(2, 0, 1)
        return img, 0
