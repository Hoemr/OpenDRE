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

Create a DataModule class for OOD detection:

```python
# In models/pl_data.py
class OutOfDistributionDetectionDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.data_dir = getattr(args, 'ood_data_dir', '../data')
        self.in_dist = getattr(args, 'ood_in_dist', 'mnist')
        self.batch_size = args.batch_size
        
    def setup(self, stage=None):
        # Load in-distribution and OOD datasets
        # This is simplified - see actual implementation for details
        pass
    
    def train_dataloader(self):
        # Return training data loader
        pass
    
    def val_dataloader(self):
        # Return validation data loader  
        pass
    
    def test_dataloader(self):
        # Return test data loader
        pass
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
