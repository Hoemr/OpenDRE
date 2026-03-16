# -*- coding: utf-8 -*-
"""Real image datasets: MNIST, CIFAR, FreyFace, etc."""
import os

import numpy as np
import torch
import torchvision.datasets as dset
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from functools import partial
from scipy.io import loadmat
from PIL import Image

from .base import DataModule


REAL_IMAGE_DATASETS = ["mnist", "svhn", "cifar10", 'celeba', 'lsun_church', 'frey_face', "fashion_mnist"]
SPECS = {
        # "mnist": (1, (0,), (1.,)),
        # "fashion_mnist": (1, (0,), (1.,)),
        # "frey_face": (1, (0.,), (1.,)),
        # "svhn": (3, (0., 0., 0.), (1., 1., 1.)),
        # "cifar10": (3, (0., 0., 0.), (1., 1., 1.)),
        # "celeba": (3, (0, 0., 0.), (1., 1., 1.)),
        
        "mnist": (1, (0.1307,), (0.3081,)),
        "fashion_mnist": (1, (0.2860,), (0.3530,)),
        "frey_face": (1, (0.58,), (0.17,)),  
        "svhn": (3, (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        "cifar10": (3, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        "celeba": (3, (0.5063, 0.4258, 0.3832), (0.3106, 0.2903, 0.2897)),
        
        "lsun_church": (3, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    }
DATA_SHAPES = {
    "mnist": (1, 28, 28),
    "fashion_mnist": (1, 28, 28),
    "frey_face": (1, 28, 20),
    "svhn": (3, 32, 32),
    "cifar10": (3, 32, 32),
    "cifar100": (3, 32, 32),
    "celeba": (3, 64, 64),
    "lsun_church": (3, 64, 64),
}


class FreyFaceDataset(Dataset):
    def __init__(self, data_dir, transform=None, train=True, val_split=0.1):
        mat_data = loadmat(os.path.join(data_dir, "frey_rawface.mat"))
        self.images = mat_data['ff'].T.reshape(-1, 28, 20)
        self.transform = transform
        num_samples = len(self.images)
        indices = np.random.permutation(num_samples)
        split = int(num_samples * (1 - val_split))
        self.indices = indices[:split] if train else indices[split:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img = self.images[self.indices[idx]]
        img = Image.fromarray(img).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, 0


def collate_fn(batch, data_type="mnist"):
    return torch.stack([item[0] for item in batch])


class RealImageDataModule(DataModule):
    def __init__(self, args):
        super().__init__(args)
        self.data_dir = args.data_dir
        self.layer_type = args.layer_type
        # 噪声仅在 pl_models 的 prepare_batch / training_step 中处理
        self.val_split = args.val_split
        if isinstance(args.gpu_id, list):
            num_gpus = len(args.gpu_id)
        else:
            num_gpus = 1
        total_cpu_cores = os.cpu_count()
        workers_per_gpu = total_cpu_cores // num_gpus
        self.num_workers = min(10, max(4, workers_per_gpu - 2))
        self.channel_size, self.normalize = self._get_dataset_specs()
        self.data_shape = DATA_SHAPES.get(self.data_type, (3, 64, 64))
        self.base_transform = self._build_base_transform()
        self.train_transform = self._build_train_transform()

    def _get_dataset_specs(self):
        return SPECS[self.data_type][0], transforms.Normalize(*SPECS[self.data_type][1:])

    def _build_base_transform(self):
        transform_list = [transforms.Resize(self.data_shape[1:])]
        if self.data_type == "frey_face":
            transform_list.insert(0, transforms.Lambda(lambda x: x.convert('L')))
        transform_list.append(transforms.ToTensor())
        transform_list.append(self.normalize)
        return transforms.Compose(transform_list)

    def _build_train_transform(self):
        if self.data_type in ["cifar10", "celeba", "lsun_church"]:
            return transforms.Compose([transforms.RandomHorizontalFlip(), self.base_transform])
        elif self.data_type == "frey_face":
            return transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(28, scale=(0.9, 1.0)),
                self.base_transform,
            ])
        return self.base_transform

    def prepare_data(self):
        if self.data_type == "mnist":
            dset.MNIST(self.data_dir, train=True, download=True)
            dset.MNIST(self.data_dir, train=False, download=True)
        elif self.data_type == "fashion_mnist":
            dset.FashionMNIST(self.data_dir, train=True, download=True)
            dset.FashionMNIST(self.data_dir, train=False, download=True)
        elif self.data_type == "svhn":
            dset.SVHN(self.data_dir, split="train", download=True)
            dset.SVHN(self.data_dir, split="test", download=True)
        elif self.data_type == "cifar10":
            dset.CIFAR10(self.data_dir, train=True, download=True)
            dset.CIFAR10(self.data_dir, train=False, download=True)
        elif self.data_type == "lsun_church":
            dset.LSUN(self.data_dir, classes=['church_outdoor_train'], download=True)
            dset.LSUN(self.data_dir, classes=['church_outdoor_val'], download=True)

    def setup(self, stage=None):
        if (stage == "fit" or stage == "validate") or stage is None:
            train_dataset = self._load_dataset(train=True)
            dataset_size = len(train_dataset)
            val_size = int(dataset_size * self.val_split)
            self.train_dataset, self.val_dataset = random_split(
                train_dataset, [dataset_size - val_size, val_size]
            )
        if stage == "test" or stage is None:
            self.test_dataset = self._load_dataset(train=False)

    def _load_dataset(self, train=True):
        transform = self.train_transform if train else self.base_transform
        if self.data_type == "mnist":
            return dset.MNIST(self.data_dir, train=train, transform=transform)
        elif self.data_type == "fashion_mnist":
            return dset.FashionMNIST(self.data_dir, train=train, transform=transform)
        elif self.data_type == "frey_face":
            return FreyFaceDataset(self.data_dir, transform=transform, train=train, val_split=self.val_split)
        elif self.data_type == "svhn":
            return dset.SVHN(self.data_dir, split="train" if train else "test", transform=transform)
        elif self.data_type == "cifar10":
            return dset.CIFAR10(self.data_dir, train=train, transform=transform)
        elif self.data_type == "celeba":
            return dset.CelebA(
                self.data_dir, split="train" if train else "test",
                transform=transforms.Compose([transforms.ToPILImage(), transform]),
            )
        elif self.data_type == "lsun_church":
            return dset.LSUN(
                self.data_dir,
                classes=['church_outdoor_train' if train else 'church_outdoor_val'],
                transform=transforms.Compose([transforms.Resize(96), transforms.RandomCrop(64), transform]),
            )

    def _create_dataloader(self, dataset, batch_size=64, shuffle=False):
        num_workers = self.num_workers
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            pin_memory=True, persistent_workers=num_workers > 1,
            collate_fn=partial(collate_fn, data_type=self.data_type),
        )
