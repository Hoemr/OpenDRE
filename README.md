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
cd nsf && pip install -e .
```

### Requirements

- Python 3.10+
- PyTorch 2.1.2+
- PyTorch Lightning 2.1.2+
- torchdiffeq, torchquad
- See `requirements.txt` for full list

## Project structure (flat packages, no `models/`)

The codebase is split into top-level packages for easier navigation:

| Package | Contents |
|---------|----------|
| `nn_models/` | Neural network models (UnifiedModel, JointScore, ResNet DRE, etc.) |
| `layers/` | Building blocks: custom layers (`layers.py`), activations (`activations.py`) |
| `polynomials/` | Polynomial basis (e.g. for score decomposition) |
| `pl_data/` | Lightning DataModules and dataset configs |
| `losses/` | Training objectives (score matching, kernel/neural DRE) |
| `estimators/` | Likelihood / density-ratio estimators, quadrature, metrics |
| `path/` | Path (ODE/SDE) and Bayesian path (path_base.py, bayesian_path.py) |
| `pl_models/` | Lightning modules per task |
| `nsf/` | NSF flows (optional submodule) |
| `config/` | Argument parsing and save paths |
| `logger/` | Logging utilities |
| `sampling/` | Sampling routines |
| `visualization/` | Plotting and visualization |
| `core/` | Project helpers (utils.py, data_utils.py, tabular/). Named `core` to avoid shadowing torchquad's `utils`. |

Import examples: `from pl_data import SUBTASK_DATA_MODULES`, `from nn_models import JointScoreModel`, `from config import parse_args`, `from path import ToyInterpXt`.

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

# Extending the Framework: Tasks and Datasets

## Pipeline

```
main.py → pl_data (DataModule) → pl_models (Lightning module) → training
```

- **main.py**: Entry point and argument parsing.
- **pl_data/**: DataModules and dataset configs; register in `SUBTASK_DATA_MODULES`.
- **pl_models/**: Lightning modules per task; register in `SUBTASK_MODLE_MODULES`.
- **config/**: Task-specific arguments (e.g. `get_fdiv_args`, `get_likelihood_args`).
- **nn_models/**: Model architectures used by `pl_models` via `build_model()`.

## Adding a New Task (Checklist)

Use an existing task (e.g. OOD detection in `pl_models/ood_detection.py`, `pl_data/ood_detection.py`) as a template.

| Step | Where | What to do |
|------|--------|------------|
| 1 | `config/config_setting.py` | Add arguments in the right `get_*_args()` (e.g. `get_fdiv_args`). |
| 2 | `pl_data/` | Add dataset name to the task’s list; implement a DataModule and register it in `SUBTASK_DATA_MODULES`. |
| 3 | `pl_models/` | Add a Lightning module (inherit from `BaseDREMetrics` or the task base); implement `build_model()`, `training_step`, and optionally `validation_step`; register in `SUBTASK_MODLE_MODULES`. |
| 4 | `main.py` | In the task branch, instantiate your DataModule and Model and pass them to the trainer. |

No need to copy large blocks: open the corresponding file for an existing subtask and mirror its structure (imports, registration keys, method names).

## Key Conventions

- **Density ratio**: Many tasks use **qx** (reference, e.g. t=0) and **px** (target, e.g. t=1). The model estimates r(x) = px(x) / qx(x). Batch format is often `(qx, px)` or `(x0, x1)`.
- **New datasets**: For synthetic data, add a generator in `core/` (e.g. `data_utils.py`) and wire it in the task’s DataModule. For tabular/image, add the name to the task list in `pl_data` and implement loading in that DataModule.
- **Custom losses**: Implement a train-step callable (e.g. `def __call__(self, model, batch, step) -> dict`) and use it in the task’s `set_basic_functions()` / `train_step_fn`. See `losses/score_matching.py`, `losses/kernel_density_ratio.py`, `losses/neural_density_ratio.py` for the expected interface.

## Model Architecture (nn_models/)

Models live in `nn_models/`: tabular in `tabular_models.py`, image in `image_models.py`, unified API in `unified.py`. Use the existing pl_models’ `build_model()` as the reference for constructor arguments.

| Use case | Tabular | Image |
|----------|---------|--------|
| Score s(x,t) | `JointScoreModel`, `DecompJointScoreModel` | `ImageJointScoreModel`, `ImageDecompJointScoreModelResNet` |
| Density ratio r(x) | `NeuralDensityRatioModel` | `ResNetDensityRatioModel` |

Example (tabular score model): `JointScoreModel(input_dim, hidden_dims=[128,128], embed_dim=256, nonlinearity="leakyrelu", args=args)`. For image backbones (ResNet, WideResNet), see `pl_models/ood_detection.py` or `pl_models/density_estimation_real_image.py` for how the pl_model builds the nn_model. Activation and polynomial options are in the docstrings and config; see `nn_models/unified.py` and `layers/activations.py` for details.

## Directory Layout

```
OS_DRE/
├── main.py
├── config/           # Argument parsing (e.g. config_setting.py)
├── pl_data/          # DataModules and dataset configs
├── pl_models/        # Lightning modules per task
├── nn_models/        # Model architectures (UnifiedModel, JointScore, ResNet DRE, etc.)
├── path/             # Path (ODE/SDE), Bayesian path
├── losses/           # Score matching, kernel/neural DRE losses
├── estimators/       # Likelihood and density-ratio estimators
├── sampling/         # Sampling routines
├── visualization/    # Plotting
├── layers/           # Custom layers, activations
├── core/             # Utils, data_utils, logger
└── nsf/              # Optional NSF flows
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
