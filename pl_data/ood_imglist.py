# -*- coding: utf-8 -*-
"""OOD detection with imglist datasets (path label per line). Data under data_dir, no extra subdirs."""
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .base import DataModule
from .ood_detection import OODDetectionDataset, ConcatDataset, DummyDataset


# Normalization (mean, std) per dataset type
OOD_IMGLIST_NORMALIZATION = {
    "cifar10": ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    "cifar100": ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    "mnist": ([0.1307], [0.3081]),
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}

# Benchmarks: compatible with OpenOOD data layout.
# - list_subdir: where txt list files live under data_dir (OpenOOD: "benchmark_imglist").
# - image_subdir: where images live; each line in a list file is "path label" with path relative to this (OpenOOD: "images_classic").
# - id / ood_train / ood_eval paths are relative to data_dir/list_subdir.
# How to get the txt files and nearood/farood: use OpenOOD's download script (see docstring below or README).
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

OOD_IMGLIST_IN_DIST_CHOICES = list(OOD_IMGLIST_BENCHMARKS.keys())


def get_ood_imglist_transforms(
    split,
    normalization_type="cifar10",
    image_size=32,
    pre_size=None,
):
    mean, std = OOD_IMGLIST_NORMALIZATION.get(normalization_type, OOD_IMGLIST_NORMALIZATION["cifar10"])
    is_train = split == "train"

    if normalization_type == "mnist":
        if is_train:
            t = transforms.Compose([
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            t = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        return t

    if pre_size and pre_size != image_size:
        resize = transforms.Resize(pre_size)
        crop = transforms.CenterCrop(image_size)
    else:
        resize = transforms.Resize((image_size, image_size))
        crop = transforms.Lambda(lambda x: x)

    if is_train:
        t = transforms.Compose([
            resize, crop,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        t = transforms.Compose([
            resize, crop,
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return t


class ImglistDataset(Dataset):
    """
    Dataset from a list file: each line is "image_path label".
    image_path is relative to data_dir (or absolute if path starts with /).
    """

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
        if self.data_dir and not image_name.startswith("/"):
            path = os.path.join(self.data_dir, image_name)
        else:
            path = image_name
        from PIL import Image
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            data = self.transform(image)
        else:
            data = image
        return {"data": data, "label": label, "index": index}


class ImglistDatasetAsTuple(Dataset):
    """Wrapper so that ImglistDataset returns (data, label) for OODDetectionDataset and DataLoader."""

    def __init__(self, imglist_dataset):
        self._ds = imglist_dataset

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, index):
        item = self._ds[index]
        return item["data"], item["label"]


def _join_data_dir(data_dir, *parts):
    """Join data_dir with zero or more path parts. Skips empty/absolute parts."""
    out = data_dir
    for p in parts:
        if not p or (isinstance(p, str) and p.startswith("/")):
            continue
        out = os.path.join(out, p)
    return out


class OODDetectionImglistDataModule(DataModule):
    """
    OOD detection DataModule using imglist files under data_dir.
    Benchmark and paths are defined in OOD_IMGLIST_BENCHMARKS (pl_data/ood_imglist.py).
    Same interface as OutOfDistributionDetectionDataModule: train_dataset.in_dist_dataset,
    get_eval_ood_datasets(stage), eval_ood_names, _get_in_dist_dataset(train=True).
    """

    def __init__(self, args):
        super().__init__(args)
        self.ood_in_dist = getattr(args, "ood_in_dist", "cifar10")
        if self.ood_in_dist not in OOD_IMGLIST_BENCHMARKS:
            raise ValueError(
                "ood_in_dist=%s not in OOD_IMGLIST_BENCHMARKS. Choices: %s"
                % (self.ood_in_dist, OOD_IMGLIST_IN_DIST_CHOICES)
            )
        self.data_dir = getattr(args, "data_dir", "../data")
        self._bench = OOD_IMGLIST_BENCHMARKS[self.ood_in_dist]
        self._id_datasets = {}
        self.eval_ood_names = []
        self.eval_ood_datasets = []
        self.num_classes = 0

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        b = self._bench
        num_classes = b["num_classes"]
        self.num_classes = num_classes
        normalization_type = b["normalization_type"]
        image_size = b["image_size"]
        pre_size = b.get("pre_size")
        list_subdir = b.get("list_subdir", "")
        image_subdir = b.get("image_subdir", "")
        image_root = _join_data_dir(self.data_dir, image_subdir)

        for split, rel_path in b["id"].items():
            imglist_pth = _join_data_dir(self.data_dir, list_subdir, rel_path)
            if not os.path.isfile(imglist_pth):
                continue
            transform = get_ood_imglist_transforms(
                split,
                normalization_type=normalization_type,
                image_size=image_size,
                pre_size=pre_size,
            )
            ds = ImglistDataset(
                imglist_pth=imglist_pth,
                data_dir=image_root,
                num_classes=num_classes,
                transform=transform,
            )
            self._id_datasets[split] = ImglistDatasetAsTuple(ds)

        id_train = self._id_datasets.get("train")
        if id_train is None:
            train_rel = b["id"].get("train", "")
            train_full = _join_data_dir(self.data_dir, list_subdir, train_rel)
            raise FileNotFoundError(
                "Imglist OOD mode needs list files (one line per 'image_path label'). "
                "Missing: %s (data_dir=%s). "
                "Use OpenOOD download script to get benchmark_imglist + images_classic (see README OOD section), or use built-in OOD (omit --ood_use_imglist)."
                % (train_full, os.path.abspath(self.data_dir))
            )

        ood_train_path = _join_data_dir(self.data_dir, list_subdir, b["ood_train"])
        if not os.path.isfile(ood_train_path):
            raise FileNotFoundError("OOD train list not found: %s" % ood_train_path)
        transform_test = get_ood_imglist_transforms(
            "test",
            normalization_type=normalization_type,
            image_size=image_size,
            pre_size=pre_size,
        )
        ood_train_ds = ImglistDataset(
            imglist_pth=ood_train_path,
            data_dir=image_root,
            num_classes=num_classes,
            transform=transform_test,
        )
        ood_for_train = ImglistDatasetAsTuple(ood_train_ds)

        self.train_dataset = OODDetectionDataset(id_train, ood_for_train, noise_std=0.0)
        self.train_dataset.in_dist_dataset = id_train

        self.val_dataset = DummyDataset(input_shape=(3, image_size, image_size))
        self.test_dataset = DummyDataset(input_shape=(3, image_size, image_size))

        self.eval_ood_names = []
        self.eval_ood_datasets = []
        for name, rel_path in b.get("ood_eval", []):
            pth = _join_data_dir(self.data_dir, list_subdir, rel_path)
            if not os.path.isfile(pth):
                continue
            ds = ImglistDataset(
                imglist_pth=pth,
                data_dir=image_root,
                num_classes=num_classes,
                transform=transform_test,
            )
            self.eval_ood_names.append(name)
            self.eval_ood_datasets.append(ImglistDatasetAsTuple(ds))

    def get_eval_ood_datasets(self, stage=None):
        return self.eval_ood_datasets

    def _get_in_dist_dataset(self, train=True, transform=None):
        key = "train" if train else "test"
        return self._id_datasets.get(key, self._id_datasets.get("train"))
