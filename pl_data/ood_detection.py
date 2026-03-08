# -*- coding: utf-8 -*-
"""OOD detection datasets and DataModule (built-in + imglist unified)."""
import os
import random
from PIL import Image

import numpy as np
import torch
import torchvision.datasets as dset
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

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
# These define the file paths for ID and OOD datasets in the OpenOOD format.
#
# ================================================================================
# DATA FLOW SUMMARY FOR CIFAR-10
# ================================================================================
#
# TRAINING (training_step):
#   - px (ID):   train_cifar10.txt (50,000 CIFAR-10 training images)
#   - qx (OOD):  val_tin.txt (TinyImageNet validation as OOD proxy)
#   - Returns: (px, qx, 0.0) pairs for density ratio estimation
#
# VALIDATION/TEST (_evaluate_ood):
#   - ID:  Test CIFAR-10 (test_cifar10.txt)
#   - OOD: Multiple datasets for evaluation:
#     - near-OOD: val (TIN-val), cifar100, tin (TIN-test)
#     - far-OOD:  mnist, svhn
#   - Computes: AUROC, FPR95, AUPR-IN, AUPR-OUT, ACC per OOD dataset
#
# ================================================================================

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
        "list_subdir": "benchmark_imglist",    # Directory containing txt files
        "image_subdir": "images_classic",       # Directory containing images
        # ===== ID DATASETS (in-distribution) =====
        # Used for training (px) and evaluation (ID reference)
        "id": {
            "train": "cifar10/train_cifar10.txt",   # 50,000 ID training images
            "val": "cifar10/val_cifar10.txt",       # ID validation (subset of train)
            "test": "cifar10/test_cifar10.txt",     # 10,000 ID test images
        },
        # ===== OOD TRAINING DATA (for training_step) =====
        # This is qx in the (px, qx) training pairs
        # - "selfgen": q from ID only via patch_shuffle / phase_rand / cutpaste_cutmix (no extra data)
        # - "random": use 300K random images (no data leakage with eval OOD)
        # - path to txt: use imglist dataset (e.g. val_tin.txt)
        "ood_train": "random",
        "ood_train_random_file": "300K_random_images.npy",  # Used when ood_train="random"
        "selfgen_grid": 4,  # Grid size for patch_shuffle when ood_train="selfgen"
        # ===== OOD EVALUATION DATA (for validation_step/test_step) =====
        # Used to compute AUROC, FPR95, etc. in _evaluate_ood()
        # Ordered by: near-OOD first, then far-OOD
        "ood_eval": [
            ("cifar100", "cifar10/test_cifar100.txt"), # CIFAR-100 test (near-OOD)
            ("tin", "cifar10/test_tin.txt"),         # TinyImageNet test (near-OOD)
            ("mnist", "cifar10/test_mnist.txt"),     # MNIST test (far-OOD)
            ("svhn", "cifar10/test_svhn.txt"),       # SVHN test (far-OOD)
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
        "ood_train": "random",
        "ood_train_random_file": "300K_random_images.npy",
        "selfgen_grid": 4,
        "ood_eval": [
            ("cifar10", "cifar100/test_cifar10.txt"),
            ("tin", "cifar100/test_tin.txt"),
            ("mnist", "cifar100/test_mnist.txt"),
            ("svhn", "cifar100/test_svhn.txt"),
        ],
    },
}

# Near-OOD vs Far-OOD classification (for reporting separate AUROC)
# - near-OOD: distributionally similar to ID (e.g., CIFAR-100 vs CIFAR-10)
# - far-OOD: distributionally different from ID (e.g., MNIST vs CIFAR-10)
OOD_IMGLIST_NEAR_FAR = {
    "cifar10": {"near": ["cifar100", "tin"], "far": ["mnist", "svhn"]},
    "cifar100": {"near": ["cifar10", "tin"], "far": ["mnist", "svhn"]},
}

OOD_IMGLIST_IN_DIST_CHOICES = list(OOD_IMGLIST_BENCHMARKS.keys())


# ---------------------------------------------------------------------------
# Self-generated OOD transforms (semantic-destroying, low-level-statistics-preserving)
# All operate on tensors (C, H, W); normalized space unless noted.
# ---------------------------------------------------------------------------

def _patch_shuffle(x, grid=4):
    """Grid shuffle: split image into grid x grid patches and shuffle patch order.
    x: (C, H, W) tensor."""
    C, H, W = x.shape
    gh, gw = min(grid, H), min(grid, W)
    ph, pw = H // gh, W // gw
    if ph < 1 or pw < 1:
        return x
    # (C, gh, ph, gw, pw) -> (C, gh*gw, ph, pw)
    patches = x.view(C, gh, ph, gw, pw).permute(0, 1, 3, 2, 4).reshape(C, gh * gw, ph, pw)
    idx = torch.randperm(gh * gw, device=x.device, dtype=torch.long)
    patches = patches[:, idx]
    return patches.view(C, gh, gw, ph, pw).permute(0, 1, 3, 2, 4).reshape(C, H, W)


def _phase_rand(x, mean, std):
    """Preserve amplitude spectrum, randomize phase in FFT. Done in [0,1] space: denorm -> fft -> renorm.
    x: (C, H, W) normalized; mean, std: (C,) or (C,1,1) or list/tuple."""
    if isinstance(mean, (list, tuple)):
        mean = torch.tensor(mean, dtype=x.dtype, device=x.device).view(-1, 1, 1)
    else:
        mean = mean.to(x.device).to(x.dtype)
        if mean.dim() == 1:
            mean = mean.view(-1, 1, 1)
    if isinstance(std, (list, tuple)):
        std = torch.tensor(std, dtype=x.dtype, device=x.device).view(-1, 1, 1)
    else:
        std = std.to(x.device).to(x.dtype)
        if std.dim() == 1:
            std = std.view(-1, 1, 1)
    # Denorm to [0, 1]
    x01 = (x * std + mean).clamp(0.0, 1.0)
    C, H, W = x01.shape
    out = x01.new_empty(x01.shape)
    for c in range(C):
        f = torch.fft.fft2(x01[c])
        amp = f.abs()
        phase = torch.angle(f)
        new_phase = (torch.rand_like(phase, device=x.device) * 2.0 * np.pi - np.pi)
        f_new = amp * torch.exp(1j * new_phase)
        rec = torch.fft.ifft2(f_new).real.clamp(0.0, 1.0)
        out[c] = rec
    # Renorm
    return (out - mean) / std


def _cutpaste_cutmix(x_in, x2):
    """Cutpaste or cutmix from another ID image: random box, paste (cutpaste) or blend (cutmix).
    x_in, x2: (C, H, W). Returns (C, H, W)."""
    C, H, W = x_in.shape
    w_max = max(1, W // 2)
    h_max = max(1, H // 2)
    w = random.randint(1, w_max)
    h = random.randint(1, h_max)
    top = random.randint(0, H - h) if H > h else 0
    left = random.randint(0, W - w) if W > w else 0
    out = x_in.clone()
    if random.random() < 0.5:
        # Cutpaste: paste patch from x2
        out[..., top : top + h, left : left + w] = x2[..., top : top + h, left : left + w]
    else:
        # Cutmix: blend in box
        lam = random.random()
        out[..., top : top + h, left : left + w] = (
            lam * x_in[..., top : top + h, left : left + w]
            + (1.0 - lam) * x2[..., top : top + h, left : left + w]
        )
    return out


def _selfgen_ood_transform(x_in, x2, norm_mean, norm_std, grid=4):
    """Choose one of patch_shuffle, phase_rand, cutpaste_cutmix at random; return x_ood."""
    choice = random.randint(0, 2)
    if choice == 0:
        return _patch_shuffle(x_in, grid=grid)
    if choice == 1:
        return _phase_rand(x_in, norm_mean, norm_std)
    return _cutpaste_cutmix(x_in, x2)

# def _selfgen_ood_transform(x_in, x2, norm_mean, norm_std, grid=4):
#     r = random.random()
#     if r < 0.60:
#         return _cutpaste_cutmix(x_in, x2)   # cross-class x2
#     elif r < 0.85:
#         return _patch_shuffle(x_in, grid=grid)
#     else:
#         return _phase_rand(x_in, norm_mean, norm_std)

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
    """
    Training dataset for OOD detection via density ratio estimation.
    
    This is the dataset used in training_step.
    
    Each sample returns: (px, qx, true_fdiv)
    - px: ID image (in-distribution)
    - qx: OOD image (out-of-distribution)
    - true_fdiv: placeholder value (always 0.0)
    
    Modes:
    - q_mode != "selfgen": OOD image (qx) is from ood_dataset (random index each time).
    - q_mode == "selfgen": q is produced from ID images via semantic-destroying transforms
      (patch_shuffle, phase_rand, cutpaste/cutmix); no external OOD data. Requires
      norm_mean, norm_std for phase_rand (denorm/renorm in [0,1]).
    """
    
    def __init__(self, in_dist_dataset, ood_dataset=None, q_mode="external", norm_mean=None, norm_std=None, selfgen_grid=4):
        self.in_dist_dataset = in_dist_dataset  # ID dataset (px)
        self.ood_dataset = ood_dataset          # OOD dataset (qx); None when q_mode="selfgen"
        self.q_mode = (q_mode.lower() if isinstance(q_mode, str) else q_mode)

        if self.q_mode == "selfgen":
            if norm_mean is None or norm_std is None:
                raise ValueError("OODDetectionDataset(q_mode='selfgen') requires norm_mean and norm_std.")
            self._norm_mean = torch.tensor(norm_mean, dtype=torch.float32).view(-1, 1, 1)
            self._norm_std = torch.tensor(norm_std, dtype=torch.float32).view(-1, 1, 1)
            self._selfgen_grid = selfgen_grid
            self._getitem_fn = self._getitem_selfgen
        else:
            self._getitem_fn = self._getitem_external

        self.num_samples = len(in_dist_dataset)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self._getitem_fn(idx)

    def _getitem_external(self, idx):
        """Get item for external OOD mode."""
        x_in, _ = self.in_dist_dataset[idx]
        ood_idx = random.randint(0, len(self.ood_dataset) - 1)
        x_ood, _ = self.ood_dataset[ood_idx]
        return x_in, x_ood, torch.tensor(0.0)
    
    def _getitem_selfgen(self, idx):
        """Get item for self-generated OOD mode."""
        x_in, _ = self.in_dist_dataset[idx]
        
        # x_ood = T(x_in, x2) with T in {patch_shuffle, phase_rand, cutpaste_cutmix}
        random_idx = random.randint(0, len(self.in_dist_dataset) - 1)
        x2, _ = self.in_dist_dataset[random_idx]
        
        # Ensure tensors for in-place / device (e.g. phase_rand uses x_in.device)
        if not isinstance(x_in, torch.Tensor):
            x_in = torch.from_numpy(np.array(x_in)).float()
        if not isinstance(x2, torch.Tensor):
            x2 = torch.from_numpy(np.array(x2)).float()
            
        x_ood = _selfgen_ood_transform(
            x_in, x2,
            norm_mean=self._norm_mean,
            norm_std=self._norm_std,
            grid=self._selfgen_grid,
        )
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
    """
    OOD Detection DataModule for density ratio estimation.
    
    ================================================================================
    DATA FLOW FOR TRAINING AND VALIDATION
    ================================================================================
    
    1. TRAINING DATA (used in training_step)
       ----------------
       - self.train_dataset: OODDetectionDataset instance
         - Returns (px, qx, true_fdiv) where:
           - px: ID image (from train split of ID dataset)
           - qx: OOD image (from train OOD dataset)
           - true_fdiv: always 0.0 (placeholder)
       - Source: _setup_imglist() or setup() creates this
       - ID data:   cifar10/train_cifar10.txt → images_classic/
       - OOD data:  cifar10/val_tin.txt → images_classic/  (TinyImageNet validation)
    
    2. VALIDATION DATA (used in validation_step → _evaluate_ood)
       ----------------
       - ID: self.train_dataset.in_dist_dataset (same as training ID)
       - OOD: self.eval_ood_datasets (list of OOD datasets for evaluation)
         - val:   cifar10/val_tin.txt (TinyImageNet validation)
         - cifar100: cifar10/test_cifar100.txt
         - tin:     cifar10/test_tin.txt
         - mnist:   cifar10/test_mnist.txt
         - svhn:    cifar10/test_svhn.txt
       - Near-OOD vs Far-OOD:
         - near: [val, cifar100, tin]  (similar distribution)
         - far:  [mnist, svhn]         (different distribution)
    
    3. TEST DATA (used in test_step → _evaluate_ood)
       ----------------
       - ID: self._get_in_dist_dataset(train=True) (same as training ID)
       - OOD: self.eval_ood_datasets (same as validation)
    
    ================================================================================
    KEY ATTRIBUTES (set by _setup_imglist)
    ================================================================================
    
    - self.train_dataset:          OODDetectionDataset for training
    - self.train_dataset.in_dist_dataset: ID dataset (used for val/test ID)
    - self.eval_ood_names:         List of OOD dataset names for evaluation
    - self.eval_ood_datasets:      List of OOD datasets for evaluation
    - self._id_datasets:           Dict of ID datasets {train, val, test}
    
    ================================================================================
    """
    
    def __init__(self, args):
        super().__init__(args)
        
        # ===== OOD Configuration =====
        self.ood_in_dist = getattr(args, 'ood_in_dist', 'mnist')
        self.ood_use_imglist = getattr(args, 'ood_use_imglist', False)
        self.ood_base_type = getattr(args, 'ood_base_type', 'universal')
        self.ood_crop_size = getattr(args, 'ood_crop_size', 12)
        self.num_workers = 8
        
        # ===== EVALUATION OOD DATASETS (for validation/test) =====
        # These are set by _setup_imglist() and used in _evaluate_ood()
        self.eval_ood_names = []      # e.g., ['val', 'cifar100', 'tin', 'mnist', 'svhn']
        self.eval_ood_datasets = []   # corresponding dataset objects
        
        # ===== ID DATASETS =====
        # Set by _setup_imglist(): {'train': ..., 'val': ..., 'test': ...}
        self._id_datasets = {}
        
        # ===== BENCHMARK CONFIG =====
        self._bench = OOD_IMGLIST_BENCHMARKS.get(self.ood_in_dist)
        self.num_classes = 0
        
        if self.ood_use_imglist and self.ood_in_dist not in OOD_IMGLIST_BENCHMARKS:
            raise ValueError(
                "ood_in_dist=%s not in OOD_IMGLIST_BENCHMARKS. Choices: %s"
                % (self.ood_in_dist, OOD_IMGLIST_IN_DIST_CHOICES)
            )

    def prepare_data(self):
        # ===== DOWNLOAD DATA IF NEEDED (non-imglist mode only) =====
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
        """
        Initialize datasets for training/validation/test.
        
        This method is called by PyTorch Lightning before training.
        For imglist mode, delegates to _setup_imglist().
        """
        if self.ood_use_imglist:
            self._setup_imglist(stage=stage)
            return
        
        # ===== NON-IMGLIST MODE (legacy) =====
        transform_train = self._get_transform(train=True)
        if (stage == "fit" or stage == "validate") or stage is None:
            # Create training dataset: pairs of (ID, OOD) images
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
            result = self._get_in_dist_dataset(train=train, transform=crop_transform)
        elif self.ood_base_type == 'general':
            ood_list = {'mnist': 'fashion_mnist', 'fashion_mnist': 'mnist', 'cifar10': 'svhn', 'cifar100': 'svhn'}
            ood_name = ood_list.get(self.ood_in_dist, 'svhn')
            result = self._get_ood_by_name(ood_name, train=train, transform=transform)
        elif self.ood_base_type == 'universal':
            oe_file = os.path.join(self.args.data_dir, "300K_random_images.npy")
            # Get normalization params based on ID dataset
            norm_type = self.ood_in_dist if self.ood_in_dist in OOD_IMGLIST_NORMALIZATION else "cifar10"
            mean, std = OOD_IMGLIST_NORMALIZATION[norm_type]
            image_size = 32  # Standard for CIFAR/MNIST
            result = OODRandomImagesDataset(oe_file, image_size=image_size, mean=mean, std=std)
        else:
            result = None
        return result

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

    def get_eval_id_dataset(self, stage='val'):
        """
        Get ID dataset for validation/test evaluation.
        
        This method returns the CORRECT ID dataset for evaluation:
        - For imglist mode: returns 'val' split for validation, 'test' split for test
        - This avoids data leakage by NOT using training data for evaluation
        
        Args:
            stage: 'val' for validation, 'test' for testing
            
        Returns:
            ID dataset for evaluation (not training data)
        """
        if self.ood_use_imglist:
            # Use val split for validation, test split for test
            # This avoids data leakage: model was trained on train split, 
            # so we should evaluate on a separate split
            if stage == 'val':
                return self._id_datasets.get('val', self._id_datasets.get('test'))
            else:
                return self._id_datasets.get('test', self._id_datasets.get('val'))
        else:
            # Non-imglist mode: use test set for evaluation
            return self._get_in_dist_dataset(train=False)

    def _setup_imglist(self, stage=None):
        """
        Initialize datasets using OpenOOD-style imglist format.
        
        This is the MAIN method for data initialization when using --ood_use_imglist True.
        
        CREATED DATASETS:
        =================
        1. self.train_dataset (for training_step):
           - Type: OODDetectionDataset
           - Returns: (px, qx, true_fdiv)
           - px: ID image from train_cifar10.txt
           - qx: OOD image from val_tin.txt (TinyImageNet validation set)
           - true_fdiv: 0.0 (placeholder)
        
        2. self._id_datasets (for validation/test ID):
           - {'train': ID train, 'val': ID val, 'test': ID test}
        
        3. self.eval_ood_datasets + self.eval_ood_names (for validation/test OOD):
           - List of OOD datasets for computing AUROC, FPR95, etc.
        
        DATA SOURCES FOR CIFAR-10:
        ==========================
        - ID train:     benchmark_imglist/cifar10/train_cifar10.txt
        - ID val:       benchmark_imglist/cifar10/val_cifar10.txt
        - ID test:      benchmark_imglist/cifar10/test_cifar10.txt
        - OOD train:    benchmark_imglist/cifar10/val_tin.txt (TinyImageNet for training pairs)
        - OOD eval:     benchmark_imglist/cifar10/test_cifar100.txt (near-OOD)
                        benchmark_imglist/cifar10/test_tin.txt (near-OOD)
                        benchmark_imglist/cifar10/test_mnist.txt (far-OOD)
                        benchmark_imglist/cifar10/test_svhn.txt (far-OOD)
        """
        b = self._bench
        num_classes = b["num_classes"]
        self.num_classes = num_classes
        normalization_type = b["normalization_type"]
        image_size = b["image_size"]
        pre_size = b.get("pre_size")
        list_subdir = b.get("list_subdir", "")
        image_subdir = b.get("image_subdir", "")
        image_root = _join_data_dir(self.args.data_dir, image_subdir)

        # ===== STEP 1: Load ID datasets (train/val/test) =====
        # These are used for:
        #   - train: training ID data (px in training_step)
        #   - val/test: ID data for evaluation metrics
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

        # Transform for OOD evaluation data (test mode: no augmentation)
        transform_test = get_ood_imglist_transforms(
            "test", normalization_type=normalization_type, image_size=image_size, pre_size=pre_size
        )
        
        # Get normalization mean/std for OOD training dataset
        norm_mean, norm_std = OOD_IMGLIST_NORMALIZATION.get(
            normalization_type, OOD_IMGLIST_NORMALIZATION["cifar10"]
        )

        # ===== STEP 2: Load OOD training data (for training_step) =====
        # This is qx in the training pairs (px, qx)
        # Supports three modes:
        #   - "selfgen": q from ID via transforms (patch_shuffle, phase_rand, cutpaste/cutmix); no extra data
        #   - "random": use 300K random images (no data leakage)
        #   - path to txt file: use imglist dataset
        ood_train_config = getattr(self.args, "ood_train", None) or b.get("ood_train", "")
        if ood_train_config == "selfgen":
            # Self-generated OOD: q = T(x_in, x2) from ID only
            self.train_dataset = OODDetectionDataset(
                id_train,
                ood_dataset=None,
                q_mode="selfgen",
                norm_mean=norm_mean,
                norm_std=norm_std,
                selfgen_grid=b.get("selfgen_grid", 4),
            )
        elif ood_train_config == "random":
            # Use random images dataset (no data leakage with eval OOD)
            # Uses memmap + direct tensor ops for fast loading without PIL
            random_file = b.get("ood_train_random_file", "300K_random_images.npy")
            random_path = os.path.join(self.args.data_dir, random_file)
            if not os.path.isfile(random_path):
                raise FileNotFoundError(
                    "Random images file not found: %s. "
                    "Download 300K random images for OOD training." % random_path
                )
            ood_for_train = OODRandomImagesDataset(
                random_path, image_size=image_size, mean=norm_mean, std=norm_std
            )
            self.train_dataset = OODDetectionDataset(id_train, ood_for_train)
        else:
            # Legacy: use imglist file
            ood_train_path = _join_data_dir(self.args.data_dir, list_subdir, ood_train_config)
            if not os.path.isfile(ood_train_path):
                raise FileNotFoundError("OOD train list not found: %s" % ood_train_path)
            ood_train_ds = ImglistDataset(
                imglist_pth=ood_train_path, data_dir=image_root, num_classes=num_classes, transform=transform_test
            )
            ood_for_train = ImglistDatasetAsTuple(ood_train_ds)
            self.train_dataset = OODDetectionDataset(id_train, ood_for_train)
        
        self.val_dataset = DummyDataset(input_shape=(3, image_size, image_size))
        self.test_dataset = DummyDataset(input_shape=(3, image_size, image_size))

        # ===== STEP 4: Load OOD evaluation datasets (for validation_step/test_step) =====
        # These are used to compute AUROC, FPR95, etc. in _evaluate_ood()
        # For CIFAR-10: val(TIN), cifar100, tin (near-OOD) + mnist, svhn (far-OOD)
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
    """
    Random images dataset for OOD training (qx in density ratio estimation).
    
    Optimized for fast initialization:
    - Uses np.load(..., mmap_mode="r") - only reads metadata at init
    - No data access in __init__ - format detection deferred to first __getitem__
    - Direct tensor operations (no PIL) for fast per-sample processing
    """
    def __init__(self, npy_file, image_size=32, mean=(0.4914,0.4822,0.4465), std=(0.2470,0.2435,0.2616)):
        # mmap: only reads .npy header (shape, dtype), not actual data
        self.data = np.load(npy_file, mmap_mode="r")
        
        self.image_size = int(image_size)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3,1,1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3,1,1)
        
        # Infer format from overall shape (no data access needed)
        # Shape is (N, H, W, C) for HWC or (N, C, H, W) for CHW
        shape = self.data.shape
        if len(shape) == 3:
            # (N, H, W) - grayscale
            self._is_chw = False
            self._is_grayscale = True
        elif len(shape) == 4:
            # Detect CHW vs HWC by comparing dim 1 and dim 3
            # CHW: dim 1 is small (1 or 3), dim 3 is large
            # HWC: dim 3 is small (1 or 3), dim 1 is large
            dim1, dim3 = shape[1], shape[3]
            if dim1 in (1, 3) and dim3 > 3:
                self._is_chw = True
            else:
                self._is_chw = False
            self._is_grayscale = (shape[3 if not self._is_chw else 1] == 1) if len(shape) == 4 else True
        else:
            raise ValueError(f"Unexpected data shape: {shape}")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img = self.data[idx]  # Lazy load from memmap

        # Handle memmap read-only buffer
        if isinstance(img, np.ndarray) and not img.flags.writeable:
            img = np.array(img, copy=True)

        t = torch.from_numpy(img)

        # Handle shape: unify to CHW format
        if t.ndim == 2:
            # HxW grayscale -> 1xHxW
            t = t.unsqueeze(0)
        elif t.ndim == 3:
            if self._is_chw:
                # Already CHW
                pass
            else:
                # HWC -> CHW
                t = t.permute(2, 0, 1)
        else:
            raise ValueError(f"Unexpected tensor shape: {tuple(t.shape)}")

        # Handle channels: ensure 3 channels (RGB)
        if t.shape[0] == 1:
            t = t.expand(3, -1, -1)  # 1xHxW -> 3xHxW
        elif t.shape[0] != 3:
            raise ValueError(f"Unexpected channel count: {t.shape[0]}")

        # ToTensor equivalent: uint8 -> float in [0, 1]
        if t.dtype == torch.uint8:
            t = t.float().div_(255.0)
        else:
            t = t.float()
            # Heuristic: if mean > 1, assume [0, 255] range
            if t.mean().item() > 1.0:
                t = t.div_(255.0)

        # Resize if needed (bilinear interpolation)
        if t.shape[1] != self.image_size or t.shape[2] != self.image_size:
            t = F.interpolate(
                t.unsqueeze(0), 
                size=(self.image_size, self.image_size),
                mode="bilinear", 
                align_corners=False
            ).squeeze(0)

        # Normalize (consistent with eval)
        t = t.sub_(self.mean).div_(self.std)

        return t, 0
