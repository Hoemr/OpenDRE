# -*- coding: utf-8 -*-
"""OOD detection datasets and DataModule (built-in + imglist unified)."""
import os
import random

import numpy as np
import torch
import torchvision.datasets as dset
from torch.utils.data import Dataset
from torchvision import transforms

from .base import DataModule

OOD_IN_DIST_DATASETS = ["mnist", "fashion_mnist", "cifar10", "cifar100"]
OOD_BASE_TYPES = ["local", "general", "universal"]

OOD_EVAL_CONFIG = {
    "mnist": {"ood_datasets": ["fashion_mnist", "emnist"]},
    "fashion_mnist": {"ood_datasets": ["mnist", "emnist"]},
    "cifar10": {"ood_datasets": ["svhn", "cifar100", "lsun", "celeba"]},
    "cifar100": {"ood_datasets": ["svhn", "cifar10", "lsun", "celeba"]},
}

# Imglist/OpenOOD layout configs.
OOD_IMGLIST_NORMALIZATION = {
    "cifar10": ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    "cifar100": ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    "mnist": ([0.1307], [0.3081]),
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}

OOD_IMGLIST_BENCHMARKS = {
    "cifar10": {
        "num_classes": 10,
        "image_size": 32,
        "normalization_type": "cifar10",
        "list_subdir": "benchmark_imglist",
        "image_subdir": "images_classic",
        "id": {
            "train": "cifar10/train_cifar10.txt",
            "val": "cifar10/val_cifar10.txt",
            "test": "cifar10/test_cifar10.txt",
        },
        "ood_train": "cifar10/val_tin.txt",
        "ood_eval": [
            ("val", "cifar10/val_tin.txt"),
            ("cifar100", "cifar10/test_cifar100.txt"),
            ("tin", "cifar10/test_tin.txt"),
            ("mnist", "cifar10/test_mnist.txt"),
            ("svhn", "cifar10/test_svhn.txt"),
        ],
    },
    "cifar100": {
        "num_classes": 100,
        "image_size": 32,
        "normalization_type": "cifar100",
        "list_subdir": "benchmark_imglist",
        "image_subdir": "images_classic",
        "id": {
            "train": "cifar100/train_cifar100.txt",
            "val": "cifar100/val_cifar100.txt",
            "test": "cifar100/test_cifar100.txt",
        },
        "ood_train": "cifar100/val_tin.txt",
        "ood_eval": [
            ("val", "cifar100/val_tin.txt"),
            ("cifar10", "cifar100/test_cifar10.txt"),
            ("tin", "cifar100/test_tin.txt"),
            ("mnist", "cifar100/test_mnist.txt"),
            ("svhn", "cifar100/test_svhn.txt"),
        ],
    },
}

OOD_IMGLIST_NEAR_FAR = {
    "cifar10": {"near": ["val", "cifar100", "tin"], "far": ["mnist", "svhn"]},
    "cifar100": {"near": ["val", "cifar10", "tin"], "far": ["mnist", "svhn"]},
}

OOD_IMGLIST_IN_DIST_CHOICES = list(OOD_IMGLIST_BENCHMARKS.keys())


def get_ood_imglist_transforms(split, normalization_type="cifar10", image_size=32, pre_size=None):
    mean, std = OOD_IMGLIST_NORMALIZATION.get(normalization_type, OOD_IMGLIST_NORMALIZATION["cifar10"])
    is_train = split == "train"

    if normalization_type == "mnist":
        if is_train:
            return transforms.Compose(
                [
                    transforms.RandomCrop(image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    if pre_size and pre_size != image_size:
        resize = transforms.Resize(pre_size)
        crop = transforms.CenterCrop(image_size)
    else:
        resize = transforms.Resize((image_size, image_size))
        crop = transforms.Lambda(lambda x: x)

    if is_train:
        return transforms.Compose(
            [resize, crop, transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
    return transforms.Compose([resize, crop, transforms.ToTensor(), transforms.Normalize(mean, std)])


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
    def __init__(self, in_dist_dataset, ood_dataset):
        self.in_dist_dataset = in_dist_dataset
        self.ood_dataset = ood_dataset
        self.num_samples = len(in_dist_dataset)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x_in, _ = self.in_dist_dataset[idx]
        ood_idx = random.randint(0, len(self.ood_dataset) - 1)
        x_ood, _ = self.ood_dataset[ood_idx]
        return x_in, x_ood, torch.tensor(0.0)


class ImglistDataset(Dataset):
    """Dataset from a list file where each line is 'image_path label'."""

    def __init__(self, imglist_pth, data_dir, num_classes, transform=None, maxlen=None):
        self.data_dir = data_dir.rstrip("/") if data_dir else ""
        self.num_classes = num_classes
        self.transform = transform
        self.maxlen = maxlen
        with open(imglist_pth) as f:
            self.imglist = [line.strip() for line in f if line.strip()]
        if maxlen is not None:
            self.imglist = self.imglist[:maxlen]

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        line = self.imglist[index]
        parts = line.split(" ", 1)
        image_name = parts[0]
        label = int(parts[1]) if len(parts) > 1 else 0
        path = os.path.join(self.data_dir, image_name) if self.data_dir and not image_name.startswith("/") else image_name
        from PIL import Image

        image = Image.open(path).convert("RGB")
        data = self.transform(image) if self.transform is not None else image
        return {"data": data, "label": label, "index": index}


class ImglistDatasetAsTuple(Dataset):
    """Wrapper so ImglistDataset returns (data, label)."""

    def __init__(self, imglist_dataset):
        self._ds = imglist_dataset

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, index):
        item = self._ds[index]
        return item["data"], item["label"]


def _join_data_dir(data_dir, *parts):
    out = data_dir
    for p in parts:
        if not p or (isinstance(p, str) and p.startswith("/")):
            continue
        out = os.path.join(out, p)
    return out


class OutOfDistributionDetectionDataModule(DataModule):
    def __init__(self, args):
        super().__init__(args)
        self.ood_in_dist = getattr(args, 'ood_in_dist', 'mnist')
        self.ood_use_imglist = getattr(args, 'ood_use_imglist', False)
        self.ood_base_type = getattr(args, 'ood_base_type', 'universal')
        self.ood_crop_size = getattr(args, 'ood_crop_size', 12)
        self.num_workers = 0
        self.eval_ood_names = []
        self.eval_ood_datasets = []
        self._id_datasets = {}
        self._bench = OOD_IMGLIST_BENCHMARKS.get(self.ood_in_dist)
        self.num_classes = 0
        if self.ood_use_imglist and self.ood_in_dist not in OOD_IMGLIST_BENCHMARKS:
            raise ValueError(
                "ood_in_dist=%s not in OOD_IMGLIST_BENCHMARKS. Choices: %s"
                % (self.ood_in_dist, OOD_IMGLIST_IN_DIST_CHOICES)
            )

    def prepare_data(self):
        if self.ood_use_imglist:
            return
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
        if self.ood_use_imglist:
            self._setup_imglist(stage=stage)
            return
        transform_train = self._get_transform(train=True)
        if (stage == "fit" or stage == "validate") or stage is None:
            in_dist_dataset = self._get_in_dist_dataset(train=True, transform=transform_train)
            ood_dataset = self._get_ood_dataset(train=True, transform=transform_train)
            self.train_dataset = OODDetectionDataset(in_dist_dataset, ood_dataset)
            self.val_dataset = DummyDataset(input_shape=(3, 32, 32))
        if stage == "test" or stage is None:
            self.test_dataset = DummyDataset(input_shape=(3, 32, 32))

    def _get_transform(self, train=True):
        # Prefer imglist transform settings when available.
        if self.ood_in_dist in ['cifar10', 'cifar100']:
            split = "train" if train else "test"
            return get_ood_imglist_transforms(split, normalization_type=self.ood_in_dist, image_size=32)
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
        if self.ood_use_imglist:
            key = "train" if train else "test"
            return self._id_datasets.get(key, self._id_datasets.get("train"))
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
        if self.ood_use_imglist:
            return self.eval_ood_datasets
        eval_ood_datasets = OOD_EVAL_CONFIG.get(self.ood_in_dist, {}).get('ood_datasets', [])
        transform_test = self._get_transform(train=False)
        return self._load_ood_benchmarks(eval_ood_datasets, train=False, transform=transform_test)

    def _setup_imglist(self, stage=None):
        b = self._bench
        num_classes = b["num_classes"]
        self.num_classes = num_classes
        normalization_type = b["normalization_type"]
        image_size = b["image_size"]
        pre_size = b.get("pre_size")
        list_subdir = b.get("list_subdir", "")
        image_subdir = b.get("image_subdir", "")
        image_root = _join_data_dir(self.args.data_dir, image_subdir)

        self._id_datasets = {}
        for split, rel_path in b["id"].items():
            imglist_pth = _join_data_dir(self.args.data_dir, list_subdir, rel_path)
            if not os.path.isfile(imglist_pth):
                continue
            transform = get_ood_imglist_transforms(
                split, normalization_type=normalization_type, image_size=image_size, pre_size=pre_size
            )
            ds = ImglistDataset(
                imglist_pth=imglist_pth, data_dir=image_root, num_classes=num_classes, transform=transform
            )
            self._id_datasets[split] = ImglistDatasetAsTuple(ds)

        id_train = self._id_datasets.get("train")
        if id_train is None:
            train_rel = b["id"].get("train", "")
            train_full = _join_data_dir(self.args.data_dir, list_subdir, train_rel)
            raise FileNotFoundError(
                "Imglist OOD mode needs list files (one line per 'image_path label'). "
                "Missing: %s (data_dir=%s). "
                "Use OpenOOD download script to get benchmark_imglist + images_classic."
                % (train_full, os.path.abspath(self.args.data_dir))
            )

        ood_train_path = _join_data_dir(self.args.data_dir, list_subdir, b["ood_train"])
        if not os.path.isfile(ood_train_path):
            raise FileNotFoundError("OOD train list not found: %s" % ood_train_path)
        transform_test = get_ood_imglist_transforms(
            "test", normalization_type=normalization_type, image_size=image_size, pre_size=pre_size
        )
        ood_train_ds = ImglistDataset(
            imglist_pth=ood_train_path, data_dir=image_root, num_classes=num_classes, transform=transform_test
        )
        ood_for_train = ImglistDatasetAsTuple(ood_train_ds)

        self.train_dataset = OODDetectionDataset(id_train, ood_for_train)
        self.train_dataset.in_dist_dataset = id_train
        self.val_dataset = DummyDataset(input_shape=(3, image_size, image_size))
        self.test_dataset = DummyDataset(input_shape=(3, image_size, image_size))

        self.eval_ood_names = []
        self.eval_ood_datasets = []
        for name, rel_path in b.get("ood_eval", []):
            pth = _join_data_dir(self.args.data_dir, list_subdir, rel_path)
            if not os.path.isfile(pth):
                continue
            ds = ImglistDataset(
                imglist_pth=pth, data_dir=image_root, num_classes=num_classes, transform=transform_test
            )
            self.eval_ood_names.append(name)
            self.eval_ood_datasets.append(ImglistDatasetAsTuple(ds))


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
