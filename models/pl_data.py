# -*- coding: utf-8 -*-
import os
import numpy as np
import random
from functools import partial
from abc import ABC, abstractmethod

import torch
import torchvision.datasets as dset
from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, IterableDataset, Dataset, random_split

from scipy.io import loadmat
from PIL import Image

from utils import data_utils

# In[]
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
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, 
                          pin_memory=False, persistent_workers=num_workers > 1)
    
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
        return DataLoader(dataset, batch_size=None, num_workers=self.num_workers, 
                          pin_memory=True, persistent_workers=num_workers > 1)

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset)

# In[] Likelihood datamodule
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
        if (stage == "fit" or stage=="validate") or stage is None:
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
        self.rng = np.random.RandomState(seed=self.seed)
        self.generator = data_utils.LikelihoodDataGenerator(data_type=self.data_type, d=2, rng=self.rng, args=args)
        
    def setup(self, stage=None):
        sampling_fn = lambda num_samples: self.generator.sample(num_samples=num_samples)
        self.train_dataset = Synthetic2DDataset(
            sampling_fn, 
            batch_size=self.batch_size, 
            batch_num= self.batch_num,
        )
        self.val_dataset = Synthetic2DDataset(
            sampling_fn,
            batch_size=self.test_batch_size,
            batch_num=1,
        )
        self.test_dataset = self.val_dataset

# In[] Image datasets
REAL_IMAGE_DATASETS = ["mnist", "svhn", "cifar10", 'celeba', 'lsun_church', 'frey_face', "fashion_mnist"]
SPECS = {
        "mnist": (1, (0,), (1.,)),
        "fashion_mnist": (1, (0,), (1.,)),
        "frey_face": (1, (0.,), (1.,)),
        "svhn": (3, (0., 0., 0.), (1., 1., 1.)),
        "cifar10": (3, (0., 0., 0.), (1., 1., 1.)),
        "celeba": (3, (0, 0., 0.), (1., 1., 1.)),
        
        # "mnist": (1, (0.1307,), (0.3081,)),
        # "fashion_mnist": (1, (0.2860,), (0.3530,)),
        # "frey_face": (1, (0.58,), (0.17,)),  
        # "svhn": (3, (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        # "cifar10": (3, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        # "celeba": (3, (0.5063, 0.4258, 0.3832), (0.3106, 0.2903, 0.2897)),
        
        "lsun_church": (3, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    }
DATA_SHAPES = {
            "mnist": (1, 28, 28),
            "fashion_mnist": (1, 28, 28),
            "frey_face": (1, 28, 20),
            "svhn": (3, 32, 32),   # (3, 32, 32)
            "cifar10": (3, 32, 32),
            "cifar100": (3, 32, 32),
            "celeba": (3, 64, 64),
            "lsun_church": (3, 64, 64)
        }

class FreyFaceDataset(Dataset):
    def __init__(self, data_dir, transform=None, train=True, val_split=0.1):
        mat_data = loadmat(os.path.join(data_dir, "frey_rawface.mat"))
        self.images = mat_data['ff'].T.reshape(-1, 28, 20)  # initial size 28x20
        self.transform = transform
        
        num_samples = len(self.images)
        indices = np.random.permutation(num_samples)
        split = int(num_samples * (1 - val_split))
        self.indices = indices[:split] if train else indices[split:]
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        img = self.images[self.indices[idx]]
        img = Image.fromarray(img).convert('L')  # to PIL grey image
        
        if self.transform:
            img = self.transform(img)
        return img, 0  # no label
    
def collate_fn(batch, data_type="mnist"):
    return torch.stack([item[0] for item in batch])

class RealImageDataModule(DataModule):
    def __init__(self, args):
        super().__init__(args)
        self.data_dir = args.data_dir
        self.layer_type = args.layer_type
        self.noise_std = args.GRBC_epsilon
        self.val_split = args.val_split
        
        if isinstance(args.gpu_id, list):
            num_gpus = len(args.gpu_id)
        else:
            num_gpus = 1
        total_cpu_cores = os.cpu_count()
        workers_per_gpu = total_cpu_cores // num_gpus
        self.num_workers = min(10, max(4, workers_per_gpu - 2))
        # self.num_workers = 4
        
        # Dataset-specific parameters
        self.channel_size, self.normalize = self._get_dataset_specs()
        self.data_shape = DATA_SHAPES.get(self.data_type, (3, 64, 64))
        
        # Data transforms
        self.base_transform = self._build_base_transform()
        self.train_transform = self._build_train_transform()
    
    def _get_dataset_specs(self):
        """Get dataset-specific parameters (channels, normalization)."""
        return SPECS[self.data_type][0], transforms.Normalize(*SPECS[self.data_type][1:])

    def _build_base_transform(self):
        """Build base transform pipeline."""
        transform_list = [
            transforms.Resize(self.data_shape[1:]),
            transforms.ToTensor(),   # to [0,1]
            # self.normalize
        ]
        if self.data_type == "frey_face":
            transform_list.insert(0, transforms.Lambda(lambda x: x.convert('L')))  # 确保灰度
        if self.noise_std > 0:
            transform_list.append(transforms.Lambda(lambda x: x + torch.randn_like(x) * self.noise_std))
        return transforms.Compose(transform_list)

    def _build_train_transform(self):
        """Build training transform pipeline with augmentations."""
        if self.data_type in ["cifar10", "celeba", "lsun_church"]:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                self.base_transform
            ])
        elif self.data_type == "frey_face":
            return transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(28, scale=(0.9, 1.0)),
                self.base_transform
            ])
        return self.base_transform

    def prepare_data(self):
        if self.data_type == "mnist":
            dset.MNIST(self.data_dir, train=True, download=True)
            dset.MNIST(self.data_dir, train=False, download=True)
        elif self.data_type == "fashion_mnist":
            dset.FashionMNIST(self.data_dir, train=True, download=True)
            dset.FashionMNIST(self.data_dir, train=False, download=True)
        elif self.data_type == "frey_face":
            pass
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
        if (stage == "fit" or stage=="validate") or stage is None:
            train_dataset = self._load_dataset(train=True)
            dataset_size = len(train_dataset)
            val_size = int(dataset_size * self.val_split)
            self.train_dataset, self.val_dataset = random_split(train_dataset, [dataset_size - val_size, val_size])
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
            return dset.CelebA(self.data_dir, split="train" if train else "test", transform=transforms.Compose([
                transforms.ToPILImage(), transform]))
        elif self.data_type == "lsun_church":
            return dset.LSUN(self.data_dir, classes=['church_outdoor_train' if train else 'church_outdoor_val'], transform=transforms.Compose([
                transforms.Resize(96), transforms.RandomCrop(64), transform]))
    
    def _create_dataloader(self, dataset, batch_size=64, shuffle=False): 
        num_workers = self.num_workers
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, 
                          pin_memory=True, persistent_workers=num_workers > 1, collate_fn=partial(collate_fn, data_type=self.data_type))

# In[] f divergence datamodule
DISTRIBUTION_SHIFT_DATASETS = ['gauss_shift', 'beta_shift', 'cifar_c', 'domainbed', 'gauss_shift_kl']
class DistributionShiftIterDataset(IterableDataset):
    def __init__(self, generator, batch_size=5000, batch_num=1, args=None, **kwargs):
        self.generator = generator#(data_type=data_type, d=d, rng=rng, args=args)
        self.batch_size = batch_size
        self.batch_num = batch_num
        
    def __iter__(self):
        for _ in range(self.batch_num):
            yield self.generator.sample(batch_size=self.batch_size)
        
class DistributionShiftIterDataModule(IterDataModule):
    def __init__(self, args):
        super().__init__(args)
        self.shift_step = 0
        self.shift = args.shift
        self.generator = None

    def setup(self, stage=None):
        rng = np.random.RandomState(seed=self.seed)
        self.generator = data_utils.DistributionShiftDatasetGenerator(
            data_type=self.data_type, d=self.d, rng=rng, args=self.args
        )
        self.train_dataset = DistributionShiftIterDataset(
            self.generator, 
            batch_size=self.batch_size, 
            batch_num=self.batch_num, 
            args=self.args
        )
        self.val_dataset = DistributionShiftIterDataset(
            self.generator, 
            batch_size=self.test_batch_size, 
            batch_num=1, 
            args=self.args
        )
        self.test_dataset = self.val_dataset

MI_DATASETS = ['half_cube_map', 'asinh_mapping', 'additive_noise', 'gamma_exponential', 'diag_multi_gauss', 'gauss']
class MutualInformationDataset(Dataset):
    def __init__(self, data_generator, rho, batch_size=5000, batch_num=1, d=2, data_type="gauss", rng=None):
        joint_data, marginal_data, true_mi = data_generator(rho=rho, batch_size=batch_size, batch_num=batch_num, d=d, data_type=data_type, rng=rng)
        self.joint_data = joint_data   # qx
        self.marginal_data = marginal_data   # px
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
        self.train_dataset = MutualInformationDataset(  # remember change the batch size
            data_utils.generate_MI_data, 
            rho=self.rho,
            batch_size=self.batch_size, 
            batch_num=self.batch_num, 
            d=self.d, 
            data_type=self.data_type,
            rng=rng
        )
        self.val_dataset = MutualInformationDataset(
            data_utils.generate_MI_data, 
            rho=self.rho,
            batch_size=self.test_batch_size, 
            batch_num=1, 
            d=self.d, 
            data_type=self.data_type,
            rng=rng
        )
        self.test_dataset = self.val_dataset
    
class MutualInformationIterDataset(IterableDataset):
    def __init__(self, data_generator, rho, batch_size=5000, batch_num=1, d=2, data_type="gauss", rng=None):
        self.batch_num = batch_num
        self.generator = partial(data_generator, rho=rho, batch_size=batch_size, batch_num=10, d=d, data_type=data_type, rng=rng)

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
            data_utils.generate_MI_data, 
            rho=self.rho,
            batch_size=self.batch_size, 
            batch_num=self.batch_num, 
            d=self.d, 
            data_type=self.data_type,
            rng=rng
        )
        self.val_dataset = MutualInformationIterDataset(
            data_utils.generate_MI_data, 
            rho=self.rho,
            batch_size=self.test_batch_size, 
            batch_num=1, 
            d=self.d, 
            data_type=self.data_type,
            rng=rng
        )
        self.test_dataset = self.val_dataset

# In[] MI real datasets
MI_REAL_DATASETS = ['image', 'text', 'mixture']
class MIRealDataset(Dataset):
    """Dataset for real-world MI estimation with fixed target MI."""
    
    def __init__(self, generator, target_mi, num_samples=10000, batch_size=64):
        self.generator = generator
        self.target_mi = target_mi
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.bsc_p = generator.cal_bsc(generator.ds, target_mi)
        
        # Pre-generate all data
        self._generate_data()
    
    def _generate_data(self):
        """Pre-generate all data samples."""
        batch_num = (self.num_samples + self.batch_size - 1) // self.batch_size
        self.data_pairs = []
        
        for _ in range(batch_num):
            x_batch, y_batch = self.generator.generate_batch(
                self.batch_size, bsc_p=self.bsc_p)
            self.data_pairs.extend(list(zip(x_batch, y_batch)))
        
        # Trim to exact num_samples
        self.data_pairs = self.data_pairs[:self.num_samples]
        
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        x, y = self.data_pairs[idx]
        return x, y, torch.tensor(self.target_mi, dtype=torch.float32)

class MIRealIterDataset(IterableDataset):
    """Iterable dataset for real-world MI estimation with streaming data."""
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
        
        # Initialize dataset generator
        self.generator = data_utils.MIRealDatasetGenerator(
            data=args.data,
            dname1=args.mi_dname1,
            dname2=args.mi_dname2,
            ds=args.mi_ds,
            dr=args.mi_dr,   # dr means the target spatial dimension: H * W
            image_channels=args.mi_image_channels,
            image_patches=eval(args.mi_image_patches),
            nuisance=args.mi_nuisance,
            data_dir=args.data_dir
        )
    
    def setup(self, stage=None):
        """Setup datasets for different stages."""
        if (stage == "fit" or stage=="validate") or stage is None:
            if self.args.mi_mode == "stepwise":
                # For stepwise training, use iterable dataset
                self.train_dataset = self._create_stepwise_dataset()
            else:
                # For single MI value training
                self.train_dataset = MIRealDataset(
                    self.generator,
                    target_mi=self.args.true_mi,
                    num_samples=self.args.mi_n_samples,
                    batch_size=self.batch_size
                )
            
            self.val_dataset = MIRealDataset(
                self.generator,
                target_mi=self.args.true_mi,
                num_samples=5000,
                batch_size=self.test_batch_size
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = self.val_dataset
            
    def _create_stepwise_dataset(self):
        """Create dataset for stepwise MI training."""
        mi_values = [2, 4, 6, 8, 10]  # Default stepwise values
        steps_per_mi = self.args.mi_n_steps // len(mi_values)
        return StepwiseMIIterDataset(self.generator, mi_values, steps_per_mi, self.batch_size)
        
    def _create_dataloader(self, dataset, batch_size=64, shuffle=False):
        """Create DataLoader with appropriate settings."""
        num_workers = self.num_workers
        if isinstance(dataset, IterableDataset):
            return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=True, persistent_workers=num_workers > 1)
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                              pin_memory=True, persistent_workers=num_workers > 1)
    
# In[] f-div : two sample test
def generate_two_sample_test_data(batch_size, d, mu0=0, sigma0=1, mu1=0, sigma1=1, data_type="gauss_two_sample", rng=None):
    if rng is None:
        rng = np.random.RandomState()
    
    mean0 = np.ones(d) * mu0
    mean1 = np.ones(d) * mu1
    cov0 = np.eye(d) * sigma0
    cov1 = np.eye(d) * sigma1

    if data_type == "gauss_two_sample":
        qx = rng.multivariate_normal(mean1, cov1, batch_size)
        px = rng.multivariate_normal(mean0, cov0, batch_size)
        
    else:
        raise ValueError(f"Unsupported distribution: {data_type}")
    
    return torch.from_numpy(qx).type(torch.float32), torch.from_numpy(px).type(torch.float32), 0.

class TwoSampleTestIterDataset(IterableDataset):
    def __init__(self, data_generator, batch_size=5000, batch_num=1, d=2, mu0=0, sigma0=1, mu1=0, sigma1=1, data_type="gauss_two_sample", rng=None):
        self.batch_num = batch_num
        self.generator = partial(data_generator, batch_size=batch_size, d=d, mu0=mu0, sigma0=sigma0, mu1=mu1, sigma1=sigma1, data_type=data_type, rng=rng)

    def __iter__(self):
        for _ in range(self.batch_num):
            qx, px, _ = self.generator()
            yield qx.type(torch.float32), px.type(torch.float32), 0.

class TwoSampleTestIterDataModule(IterDataModule):
    def __init__(self, args):
        super().__init__(args)
        self.mu0 = args.mu0
        self.sigma0 = args.sigma0
        self.mu1 = args.mu1
        self.sigma1 = args.sigma1

    def setup(self, stage=None):
        rng = np.random.RandomState(seed=self.seed)
        self.train_dataset = TwoSampleTestIterDataset(  
            generate_two_sample_test_data, 
            batch_size=self.batch_size, 
            batch_num=self.batch_num, 
            d=self.d, 
            mu0=self.mu0, sigma0=self.sigma0, 
            mu1=self.mu1, sigma1=self.sigma1,
            data_type=self.data_type,
            rng=rng
        )
        self.val_dataset = TwoSampleTestIterDataset(
            generate_two_sample_test_data, 
            batch_size=self.test_batch_size, 
            batch_num=1, 
            d=self.d, 
            mu0=self.mu0, sigma0=self.sigma0,
            mu1=self.mu1, sigma1=self.sigma1,
            data_type=self.data_type,
            rng=rng
        )
        self.test_dataset = self.val_dataset

# In[] f-div : change point detection
CHANGEPOINT_DATASETS = ['cpd_mc', 'cpd_text', 'cpd_creditcard']

class DREPairDataset(Dataset):
    """
    Dataset that returns (px, qx) pairs for DRE training.
    Each call to __getitem__ generates a new batch via the generator.
    """
    def __init__(self, generator, num_samples: int, batch_size: int):
        self.generator = generator
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_batches = num_samples // batch_size

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        px, qx = self.generator.sample(batch_size=self.batch_size)
        # true MI or f-div is not used in CPD, so return dummy label
        return px, qx, torch.tensor(0.0)


class FullSequenceDataset(Dataset):
    """
    Dataset that returns the full time series for CPD evaluation.
    Used in validation and test stages.
    """
    def __init__(self, generator, data_type: str):
        if data_type == "cpd_mc":
            # For synthetic, generate a long sequence with a change point
            self.data = generator.sample_monte_carlo(
                num_samples=2000, d=generator.d, experiment_id=999
            )
        elif data_type == "cpd_text":
            self.data = generator.load_and_process_text_data()
        elif data_type == "cpd_creditcard":
            self.data = generator.load_and_process_creditcard_data()
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

    def __len__(self):
        return 1  # One full sequence per dataset

    def __getitem__(self, idx):
        return self.data  # shape: (T, d)

class ChangePointDetectionDataModule(DataModule): 
    def __init__(self, args):
        super().__init__(args)
        self.num_workers = 0  # Avoid multiprocessing issues with generator
        self.batch_size = args.batch_size
        self.num_samples = getattr(args, 'num_samples', 10000)
        self.d = getattr(args, 'd', 6)
        self.seed = args.seed
        
        self.generator = data_utils.ChangePointDetectionDataGenerator(
            data_type=self.data_type, d=self.d, rng=None, args=args
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # Training: i.i.d. (px, qx) pairs
            self.train_dataset = DREPairDataset(generator=self.generator, num_samples=self.num_samples, batch_size=self.batch_size)

        if stage == "validate" or stage is None:
            # Validation: full sequence for CPD
            self.val_dataset = FullSequenceDataset(generator=self.generator, data_type=self.data_type)

        if stage == "test" or stage is None:
            # Test: full sequence for CPD
            self.test_dataset = FullSequenceDataset(generator=self.generator, data_type=self.data_type)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=None,  # DREPairDataset returns full batches
            num_workers=self.num_workers,pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, num_workers=0, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=0, pin_memory=True )


# In[] OOD Detection Dataset
OOD_IN_DIST_DATASETS = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']
OOD_BASE_TYPES = ['local', 'general', 'universal']

# Define standard OOD benchmark datasets for evaluation
OOD_EVAL_CONFIG = {
    'mnist': {'ood_datasets': ['fashion_mnist', 'emnist']},
    'fashion_mnist': {'ood_datasets': ['mnist', 'emnist']},
    'cifar10': {'ood_datasets': ['svhn', 'cifar100', 'lsun', 'celeba']},
    'cifar100': {'ood_datasets': ['svhn', 'cifar10', 'lsun', 'celeba']},
}

class ConcatDataset(Dataset):
    """Concatenate multiple datasets into a single dataset."""
    
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
        # Find which dataset the index belongs to
        dataset_idx = 0
        for i, cumulative_size in enumerate(self.cumulative_sizes):
            if idx < cumulative_size:
                dataset_idx = i
                break
        
        # Calculate the actual index within that dataset
        if dataset_idx == 0:
            actual_idx = idx
        else:
            actual_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        
        return self.datasets[dataset_idx][actual_idx]

class OODDetectionDataset(Dataset):
    """
    Dataset for OOD detection with paired (x_in, x_ood) samples.
    Used for training density ratio estimators for OOD detection.
    """
    def __init__(self, in_dist_dataset, ood_dataset, noise_std=0.0):
        self.in_dist_dataset = in_dist_dataset
        self.ood_dataset = ood_dataset
        self.noise_std = noise_std
        # Use the in-distribution dataset length to ensure we can iterate through all in-dist samples
        self.num_samples = len(in_dist_dataset)
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x_in, _ = self.in_dist_dataset[idx]
        # Randomly sample from OOD dataset to ensure full utilization of OOD data
        ood_idx = random.randint(0, len(self.ood_dataset) - 1)
        x_ood, _ = self.ood_dataset[ood_idx]

        # Add Gaussian noise to both in-distribution and OOD samples
        if self.noise_std > 0:
            x_in = x_in + torch.randn_like(x_in) * self.noise_std
            x_ood = x_ood + torch.randn_like(x_ood) * self.noise_std

        return x_in, x_ood, torch.tensor(0.0)

class OutOfDistributionDetectionDataModule(DataModule):
    """
    DataModule for Out-of-Distribution (OOD) detection.
    
    Supports training density ratio estimators for OOD detection using:
        - 'local': Cropped-resized in-distribution samples as synthetic OOD
        - 'general': Different dataset as OOD (e.g., CIFAR-10 vs SVHN)
        - 'universal': 300K random images (Outlier Exposure)
    
    Training returns (x_in, x_ood) pairs where:
        - x_in: In-distribution samples
        - x_ood: Out-of-distribution samples
    
    Usage:
        python main.py fdiv_estimation --subtask ood_detection --ood_in_dist mnist --ood_base_type universal
    """
    def __init__(self, args):
        super().__init__(args)
        self.ood_in_dist = getattr(args, 'ood_in_dist', 'mnist')
        self.ood_base_type = getattr(args, 'ood_base_type', 'universal')
        self.ood_noise_std = getattr(args, 'ood_noise_std', 0.1)
        self.ood_anneal_alpha_init = getattr(args, 'ood_anneal_alpha_init', 0.0)
        self.ood_crop_size = getattr(args, 'ood_crop_size', 12)
        
        # Set num_workers to 0 for simplicity
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
            # Get in-distribution dataset
            in_dist_dataset = self._get_in_dist_dataset(train=True, transform=transform_train)
            ood_dataset = self._get_ood_dataset(train=True, transform=transform_train)
            self.train_dataset = OODDetectionDataset(
                in_dist_dataset,
                ood_dataset,
                noise_std=self.ood_noise_std,
            )
            
            # Validation: use test set for in-dist, and standard OOD evaluation benchmarks
            self.val_dataset = DummyDataset(input_shape=(3, 32, 32))
            # in_dist_val = self._get_in_dist_dataset(train=False, transform=transform_test)
            # ood_val = self._get_eval_ood_dataset(train=False, transform=transform_test)
            # self.val_dataset = OODDetectionDataset(in_dist_val, ood_val)
        
        if stage == "test" or stage is None:
            self.test_dataset = DummyDataset(input_shape=(3, 32, 32))
    
    def _get_transform(self, train=True):
        """Get transform for in-distribution dataset."""
        if self.ood_in_dist in ['mnist', 'fashion_mnist']:
            transform_list = [
                transforms.Resize(32),
                transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            ]
        else:  # CIFAR
            transform_list = [
                transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
            ]
        
        return transforms.Compose(transform_list)
    
    def _get_crop_transform(self):
        """Get crop transform for local OOD detection."""
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
        """Get in-distribution dataset."""
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
        """Get OOD dataset based on base_type."""
        if self.ood_base_type == 'local':
            # Local: cropped-resized in-distribution as synthetic OOD
            crop_transform = self._get_crop_transform()
            return self._get_in_dist_dataset(train=train, transform=crop_transform)
        
        elif self.ood_base_type == 'general':
            # General: first OOD dataset in the list
            ood_list = {
                'mnist': 'fashion_mnist',
                'fashion_mnist': 'mnist',
                'cifar10': 'svhn',
                'cifar100': 'svhn',
            }
            ood_name = ood_list.get(self.ood_in_dist, 'svhn')
            return self._get_ood_by_name(ood_name, train=train, transform=transform)
        
        elif self.ood_base_type == 'universal':
            # Universal: 300K random images
            oe_file = os.path.join(self.args.data_dir, "300K_random_images.npy")
            return OODRandomImagesDataset(oe_file, transform=transform)
    
    def _get_ood_by_name(self, ood_name, train=True, transform=None):
        """Get OOD dataset by name."""
        # Use provided transform or default to the one based on train flag
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
            return dset.CelebA(root=self.args.data_dir, split="test" if not train else "train", transform=transform, download=False,)

    def _load_ood_benchmarks(self, ood_names, train=False, transform=None):
        """Load OOD benchmark datasets with error handling."""
        loaded_datasets = []
        for ood_name in ood_names:
            try:
                # Pass the transform directly to _get_ood_by_name to ensure it's applied correctly
                ood_dataset = self._get_ood_by_name(ood_name, train=train, transform=transform)
                # Only append if the dataset was successfully loaded (not None)
                if ood_dataset is not None:
                    loaded_datasets.append(ood_dataset)
            except Exception as e:
                print(f"Warning: Could not load OOD dataset '{ood_name}': {e}")
                continue
        return loaded_datasets

    def _get_eval_ood_dataset(self, train=False, transform=None):
        """Get evaluation OOD dataset based on OOD_EVAL_CONFIG."""
        # Dataset Name: Get the list of OOD datasets for the current in-distribution dataset
        eval_ood_datasets = OOD_EVAL_CONFIG.get(self.ood_in_dist, {}).get('ood_datasets', [])
        
        # Dataset: Load the OOD benchmark datasets
        loaded_datasets = self._load_ood_benchmarks(eval_ood_datasets, train=train, transform=transform)
        
        # If no datasets were successfully loaded, fall back to the original OOD dataset
        if not loaded_datasets:
            print(f"Warning: No evaluation OOD datasets could be loaded for '{self.ood_in_dist}'. "
                  f"Falling back to original OOD dataset.")
            return self._get_ood_dataset(train=train, transform=transform)
        
        # Concatenate all successfully loaded OOD datasets
        if len(loaded_datasets) == 1:
            return loaded_datasets[0]
        else:
            return ConcatDataset(loaded_datasets)

    def get_eval_ood_datasets(self, stage='val'):
        """Get list of evaluation OOD datasets for separate AUROC calculation."""
        # Get the list of OOD datasets for the current in-distribution dataset
        eval_ood_datasets = OOD_EVAL_CONFIG.get(self.ood_in_dist, {}).get('ood_datasets', [])
        
        # Transform to use for evaluation
        transform_test = self._get_transform(train=False)
        
        # Load the OOD benchmark datasets
        loaded_datasets = self._load_ood_benchmarks(eval_ood_datasets, train=False, transform=transform_test)

        return loaded_datasets


class DummyDataset(Dataset):
    """
    A minimal dataset containing a single dummy sample.
    Used to ensure PyTorch Lightning enters the validation/test phase 
    and triggers epoch_end hooks without loading actual data.
    """
    def __init__(self, input_shape=(3, 32, 32)):
        """
        Initialize the dummy dataset.
        """
        self.input_shape = input_shape

    def __len__(self):
        # Return 1 to ensure exactly one batch is processed
        return 1

    def __getitem__(self, idx):
        # Return a zero-filled tensor and a dummy label
        dummy_img = torch.zeros(self.input_shape)
        dummy_label = 0
        return dummy_img, dummy_label


class OODRandomImagesDataset(Dataset):
    """Dataset for universal OOD detection using random images."""
    
    def __init__(self, npy_file, transform=None):
        self.data = np.load(npy_file)
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img = self.data[idx]
        
        # Handle different data formats
        if isinstance(img, np.ndarray):
            # Data is in [N, H, W, C] format with values [0, 255]
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            # Convert to tensor
            img = torch.from_numpy(img).float() / 255.0
            if img.dim() == 2:  # Grayscale [H, W]
                img = img.unsqueeze(0).repeat(3, 1, 1)
            elif img.dim() == 3 and img.shape[0] != 3:  # [H, W, C] format
                img = img.permute(2, 0, 1)
        
        return img, 0


# In[]
SUBTASK_DATA_MODULES = {
    "synthetic2d": Synthetic2DIterDataModule,
    "tabular": TabularDataModule,
    "real_image": RealImageDataModule,
    "distribution_shift": DistributionShiftIterDataModule,
    "mutual_information": MutualInformationIterDataModule,
    "mi_realdata": MIRealDataModule,
    "two_sample_test": TwoSampleTestIterDataModule,
    "change_point_detection":ChangePointDetectionDataModule,
    "ood_detection": OutOfDistributionDetectionDataModule,
}
