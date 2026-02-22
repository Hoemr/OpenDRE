# OS_DRE: A Unified Framework for Score-based Density Ratio Estimation

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository provides a unified implementation of score-based density ratio estimation methods, integrating multiple state-of-the-art papers from top-tier machine learning venues (NeurIPS, ICML, ICLR). The framework supports likelihood estimation, f-divergence estimation, mutual information estimation, change point detection, and out-of-distribution (OOD) detection.

## Overview

Score-based generative modeling has emerged as a powerful paradigm for learning complex probability distributions. This framework unifies multiple approaches:

- **Score Matching Methods**: DRE∞, D3RE, OS-DRE, ISA-DRE
- **Kernel Methods**: Kernel Density Ratio Estimation
- **Neural Methods**: KULSIF, NCE, InfoNCE, RKL, Hellinger, KL, Gamma, EW, PW
- **Continuous Normalizing Flows**: ODE-based likelihood computation
- **Image Backbones**: ResNet-18/34/50, WideResNet-28-10 for image-based tasks

### Key Features

- 🚀 **Unified Interface**: Single entry point (`main.py`) for all tasks and methods
- 📊 **Comprehensive Benchmarks**: Synthetic 2D, tabular, image datasets
- 🔧 **Flexible Configuration**: Extensive hyperparameter control via command-line args
- 📈 **Multi-Task Support**: Likelihood estimation, f-divergence, MI, CPD, OOD detection
- 🖼️ **Image Backbones**: Support for ResNet and WideResNet architectures

## Installation

```bash
# Create conda environment
conda create -n os_dre python=3.10
conda activate os_dre

# Install dependencies
pip install -r requirements.txt

# For NSF flows (optional, for real image experiments)
cd models/nsf && pip install -e .
```

### Requirements

- Python 3.10+
- PyTorch 2.1.2+
- PyTorch Lightning 2.1.2+
- torchdiffeq, torchquad
- See `requirements.txt` for full list

## Quick Start

### Command Line Interface

All experiments are run through `main.py` with subparsers for different tasks:

```bash
# Likelihood estimation
python main.py likelihood_estimation --subtask synthetic2d --data checkerboard --method dre --decomp --K_decomp 400

# Tabular density estimation
python main.py likelihood_estimation --subtask tabular --data bsds300 --method dre --decomp --K_decomp 400 --batch_size 20000 --gpu_id 1

# Real image modeling
python main.py likelihood_estimation --subtask real_image --data mnist --method dre --decomp --K_decomp 1000 --gpu_id 0

# Distribution shift estimation
python main.py fdiv_estimation --subtask distribution_shift --data gauss_shift --decomp --K_decomp 300 --shift_steps 5 --shift 0.05 --d 80

# Mutual information estimation
python main.py fdiv_estimation --subtask mutual_information --data diag_multi_gauss --decomp --K_decomp 1000 --epochs 40000 --d 160 --gpu_id 0

# Change point detection
python main.py fdiv_estimation --subtask change_point_detection --data cpd_text --use_simplex True --cpd_swl 64

# OOD detection (with ResNet-18)
python main.py fdiv_estimation --subtask ood_detection --ood_in_dist mnist --image_backbone resnet18 --epochs 100 --batch_size 512

# OOD detection (with WideResNet-28-10, recommended for CIFAR)
python main.py fdiv_estimation --subtask ood_detection --ood_in_dist cifar10 --image_backbone wideresnet28_10 --epochs 100 --batch_size 512
```

## Supported Tasks and Datasets

### 1. Likelihood Estimation

| Subtask | Datasets | Description |
|---------|----------|-------------|
| `synthetic2d` | swissroll, 8gaussians, pinwheel, circles, moons, 2spirals, checkerboard, rings, tree | 2D synthetic distributions for visualization |
| `tabular` | bsds300 (63D), power (6D), gas (8D), hepmass (21D), miniboone (43D) | UCI benchmark datasets |
| `real_image` | mnist, fashion_mnist, cifar10, svhn, celeba, lsun_church, frey_face | Image density estimation |

### 2. F-Divergence Estimation

| Subtask | Datasets | Description |
|---------|----------|-------------|
| `distribution_shift` | gauss_shift, beta_shift, cifar_c, domainbed, gauss_shift_kl | Measure distribution shift between domains |
| `mutual_information` | half_cube_map, asinh_mapping, additive_noise, gamma_exponential, diag_multi_gauss, gauss | Estimate mutual information I(X;Y) |
| `mi_realdata` | image, text, mixture | Real-world MI estimation (MNIST, IMDB, CIFAR-10) |
| `change_point_detection` | cpd_mc, cpd_text, cpd_creditcard | Detect change points in time series |
| `ood_detection` | mnist, fashion_mnist, cifar10, cifar100 | Detect out-of-distribution samples |

## Method Configuration

### Score Estimation Methods (`--sub_method score`)

| Combination Type | Description | Flag |
|-----------------|-------------|------|
| `VPSDE` | Variance Preserving SDE | Default |
| `subVPSDE` | Sub-Variance Preserving SDE | `--combination_type subVPSDE` |
| `bridge` | Bridge Matching | `--bridge` |
| `gpp` | Gaussian Process Prior | `--gpp` |
| `OT` | Optimal Transport | `--OT` |
| `SAI` | Secant Alignment Identity | `--SAI` |

### Decomposition Options

Enable score decomposition for scalable estimation:

```bash
# Enable decomposition with polynomial basis
python main.py likelihood_estimation --subtask tabular --data bsds300 --decomp --K_decomp 400 --polynomials gaussianrbf

# Polynomial options: gaussianrbf, imq, linear, polynomial, hermite, legendre
```

### Neural/Kernel Methods (`--sub_method neural|kernel`)

```bash
# Neural density ratio estimation
python main.py likelihood_estimation --subtask synthetic2d --sub_method neural --subsub_method kulsif

# Kernel density ratio estimation
python main.py likelihood_estimation --subtask synthetic2d --sub_method kernel
```

### Image Backbone Options (for image-based tasks)

```bash
# Use ResNet-18 backbone (default for MNIST-like datasets)
python main.py fdiv_estimation --subtask ood_detection --ood_in_dist mnist --image_backbone resnet18

# Use ResNet-50 backbone
python main.py fdiv_estimation --subtask ood_detection --ood_in_dist mnist --image_backbone resnet50

# Use WideResNet-28-10 (recommended for CIFAR-10/100)
python main.py fdiv_estimation --subtask ood_detection --ood_in_dist cifar10 --image_backbone wideresnet28_10 --image_dropout 0.3
```

## Key Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | Required | `likelihood_estimation` or `fdiv_estimation` |
| `--subtask` | Required | Task-specific subtask |
| `--method` | `dre` | `dre` or `cnf` |
| `--sub_method` | `score` | `score`, `neural`, or `kernel` |
| `--subsub_method` | `dre_infty` | Specific method variant |
| `--data` | Required | Dataset name |
| `--gpu_id` | `[0]` | GPU device IDs |
| `--batch_size` | 9600 | Training batch size |
| `--epochs` | 3000 | Number of training epochs |
| `--lr` | 1e-3 | Learning rate |
| `--dims` | `128-128-128-128-128` | Hidden layer dimensions |
| `--K_decomp` | 400 | Number of decomposition components |
| `--decomp` | False | Enable score decomposition |
| `--seed` | 3407 | Random seed |

### OOD Detection Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--ood_in_dist` | `mnist` | In-distribution dataset |
| `--ood_base_type` | `universal` | Base distribution type: local, general, universal |
| `--ood_noise_std` | 0.1 | Noise standard deviation |
| `--ood_data_dir` | `../data` | Data directory |
| `--image_backbone` | `resnet18` | Image backbone: resnet18, resnet34, resnet50, wideresnet28_10 |
| `--image_dropout` | 0.3 | Dropout rate for WideResNet |

---

# Tutorial: Adding New Tasks and Datasets

This section provides a step-by-step guide to adding new tasks (e.g., OOD detection) and datasets to OS_DRE.

## Overview of Architecture

OS_DRE uses a modular design with three main components:

```
main.py → pl_data (DataModule) → pl_models (Model) → training
```

1. **main.py**: Entry point, parses arguments and coordinates everything
2. **pl_data.py**: Defines datasets and DataModule (data loading)
3. **pl_models.py**: Defines PyTorch Lightning models (training/validation)
4. **config_setting.py**: Defines task-specific arguments
5. **base_models.py**: Neural network architectures

## Tutorial: Adding OOD Detection (Complete Example)

This is a real example of how OOD detection was added to OS_DRE.

### Step 1: Add Dataset to pl_data.py

First, add the dataset name to the appropriate list:

```python
# In models/pl_data.py
OOD_DETECTION_DATASETS = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']
```

### Step 2: Add Arguments to config_setting.py

Add task-specific arguments in the `get_fdiv_args` function:

```python
# In utils/config_setting.py
def get_fdiv_args(parser):
    # ... existing code ...
    
    # OOD detection arguments
    parser.add_argument('--ood_in_dist', type=str, default='mnist', 
                       choices=['mnist', 'fashion_mnist', 'cifar10', 'cifar100'],
                       help='In-distribution dataset for OOD detection')
    parser.add_argument('--ood_base_type', type=str, default='universal',
                       choices=['local', 'general', 'universal'],
                       help='Base distribution type for OOD detection')
    
    # Generic image backbone (shared across image tasks)
    parser.add_argument('--image_backbone', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50', 'wideresnet28_10'],
                       help='Backbone network for image-based tasks')
    parser.add_argument('--image_dropout', type=float, default=0.3,
                       help='Dropout rate for WideResNet')
    
    return parser
```

### Step 3: Add DataModule to pl_data.py

Create a Dataset class and DataModule class for OOD detection:

```python
# In models/pl_data.py
class OODDetectionDataset(Dataset):
    """
    Dataset for OOD detection with paired (x_in, x_ood) samples.
    Used for training density ratio estimators for OOD detection.
    """
    def __init__(self, in_dist_dataset, ood_dataset):
        self.in_dist_dataset = in_dist_dataset
        self.ood_dataset = ood_dataset
        # Use the smaller length to ensure paired batches
        self.num_samples = min(len(in_dist_dataset), len(ood_dataset))
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Get in-distribution sample
        x_in, _ = self.in_dist_dataset[idx]
        # Get OOD sample
        x_ood, _ = self.ood_dataset[idx]
        return x_in, x_ood

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
        # Download datasets if needed
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
        # Get transforms
        transform_train = self._get_transform(train=True)
        transform_test = self._get_transform(train=False)
        
        # Get in-distribution dataset
        in_dist_dataset = self._get_in_dist_dataset(train=True, transform=transform_train)
        
        # Get OOD dataset
        ood_dataset = self._get_ood_dataset(train=True, transform=transform_train)
        
        if (stage == "fit" or stage == "validate") or stage is None:
            self.train_dataset = OODDetectionDataset(in_dist_dataset, ood_dataset)
            
            # Validation: use test set for in-dist, and appropriate OOD
            in_dist_val = self._get_in_dist_dataset(train=False, transform=transform_test)
            ood_val = self._get_ood_dataset(train=False, transform=transform_test)
            self.val_dataset = OODDetectionDataset(in_dist_val, ood_val)
        
        if stage == "test" or stage is None:
            # Test: use test set
            in_dist_test = self._get_in_dist_dataset(train=False, transform=transform_test)
            ood_test = self._get_ood_dataset(train=False, transform=transform_test)
            self.test_dataset = OODDetectionDataset(in_dist_test, ood_test)
    
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
            return self._get_ood_by_name(ood_name, train=train)
        
        elif self.ood_base_type == 'universal':
            # Universal: 300K random images
            oe_file = os.path.join(self.args.data_dir, "300K_random_images.npy")
            return OODRandomImagesDataset(oe_file)
    
    def _get_ood_by_name(self, ood_name, train=True):
        """Get OOD dataset by name."""
        if ood_name == 'mnist':
            return dset.MNIST(self.args.data_dir, train=train, download=False, 
                            transform=self._get_transform(train))
        elif ood_name == 'fashion_mnist':
            return dset.FashionMNIST(self.args.data_dir, train=train, download=False,
                                    transform=self._get_transform(train))
        elif ood_name == 'cifar10':
            return dset.CIFAR10(self.args.data_dir, train=train, download=False,
                               transform=self._get_transform(train))
        elif ood_name == 'cifar100':
            return dset.CIFAR100(self.args.data_dir, train=train, download=False,
                                transform=self._get_transform(train))
        elif ood_name == 'svhn':
            return dset.SVHN(self.args.data_dir, split='train' if train else 'test',
                            download=False, transform=self._get_transform(train))
        elif ood_name == 'lsun':
            lsun_path = os.path.join(self.args.data_dir, 'LSUN_resize')
            return dset.LSUN(lsun_path, transform=self._get_transform(train))
        elif ood_name == 'celeba':
            return dset.CelebA(
                root=self.args.data_dir,
                split="test" if not train else "train",
                transform=transforms.Compose([transforms.Resize([32, 32]), transforms.ToTensor()]),
                download=False,
            )


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
```

### Step 4: Add Model Class to pl_models.py

Create a model class that inherits from `DensityRatioEstimationModel`:

```python
# In models/pl_models.py
class OutOfDistributionDetectionModel(DensityRatioEstimationModel):
    def __init__(self, args, save_path, datamodule):
        super().__init__(args, save_path, datamodule)
        self.ood_in_dist = getattr(args, 'ood_in_dist', 'mnist')
        self.image_backbone = getattr(args, 'image_backbone', 'resnet18')
        
    def build_model(self):
        # Build ResNet/WideResNet or default model
        image_backbone = getattr(self.args, 'image_backbone', None)
        if image_backbone in ['resnet18', 'resnet34', 'resnet50', 'wideresnet28_10']:
            self.model = base_models.ResNetDensityRatioModel(
                in_channels=3,
                num_classes=1,
                backbone=image_backbone,
                dropout=getattr(self.args, 'image_dropout', 0.3)
            )
        else:
            # Use default neural network
            pass
    
    def training_step(self, batch, batch_idx):
        # OOD-specific training logic
        px, qx, _ = self.prepare_batch(batch)
        loss_dict = self.train_step_fn(self.model, (qx, px), step=self.global_step)
        return loss_dict["loss"]
    
    def validation_step(self, batch, batch_idx):
        # Compute and log OOD metrics
        pass
```

### Step 5: Register the Model in SUBTASK_MODLE_MODULES

```python
# In models/pl_models.py
SUBTASK_MODLE_MODULES = {
    # ... existing entries ...
    "ood_detection": OutOfDistributionDetectionModel,
}
```

### Step 6: Register DataModule in main.py

```python
# In main.py
from models.pl_data import (
    # ... existing imports ...
    OutOfDistributionDetectionDataModule
)

# In the main() function
if args.subtask == "ood_detection":
    datamodule = OutOfDistributionDetectionDataModule(args)
```

## Key Patterns

### 1. Understanding the Density Ratio Framework

Most tasks in OS_DRE are based on density ratio estimation:
- **px** (t=1): Target distribution (e.g., in-distribution for OOD)
- **qx** (t=0): Reference distribution (e.g., noise or OOD samples)
- The model learns to estimate r(x) = px(x) / qx(x)

### 2. Adding New Datasets

For simple synthetic datasets:

```python
# In utils/data_utils.py
class NewDatasetGenerator(Generator):
    def generate(self, n_samples, **kwargs):
        # Generate samples from your distribution
        return samples
```

For real datasets, add to the appropriate list in `pl_data.py` and implement the loading logic.

### 3. Custom Loss Functions

If you need a custom loss, add it to `models/losses.py`:

```python
class CustomDensityRatioTrainStepFn:
    def __init__(self, args, sde, method):
        self.args = args
        self.sde = sde
        self.method = method
        
    def __call__(self, model, batch, step):
        qx, px = batch
        # Your custom training logic
        return {"loss": loss, "nll": nll}
```

## Architecture

```
OS_DRE/
├── main.py                    # Unified entry point
├── models/
│   ├── base_models.py         # Core neural network architectures
│   ├── SDE.py                 # SDE/ODE solvers (VPSDE, subVPSDE, Bridge)
│   ├── estimators.py          # Likelihood & density ratio estimators
│   ├── losses.py              # Training loss functions
│   ├── ncsn_unet.py          # Score-based UNet architecture
│   ├── layers.py              # Neural network layers (including ResNet, WideResNet)
│   ├── pl_models.py           # PyTorch Lightning wrappers
│   ├── pl_data.py             # Data modules
│   └── nsf/                   # Neural Spline Flows integration
├── utils/
│   ├── config_setting.py      # Argument parsing & configuration
│   ├── data_utils.py          # Data generators
│   ├── sampling.py            # Sampling methods (HMC, AIS)
│   ├── visualization.py       # Plotting utilities
│   └── logger.py              # Logging utilities
└── results/                   # Experiment outputs
```

## Citation

This framework integrates methods from multiple papers. Please cite the respective works:

### Core Methods

- **DRE∞**: Score-Based Generative Modeling through Stochastic Differential Equations
- **D3RE**: Density Ratio Estimation via Denoising
- **OS-DRE**: Optimal Score-Based Density Ratio Estimation
- **ISA-DRE**: Interval Annealing for Score-Based DRE

### Applications

- **CNF**: Neural Ordinary Differential Equations
- **SDE**: Score-Based Generative Modeling

## License

MIT License - See LICENSE file for details.

## Acknowledgments

This repository builds upon open-source implementations from the research community.
