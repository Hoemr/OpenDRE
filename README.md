# OS_DRE: A Unified Framework for Score-based Density Ratio Estimation

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository provides a unified implementation of score-based density ratio estimation methods, integrating multiple state-of-the-art papers from top-tier machine learning venues (NeurIPS, ICML, ICLR). The framework supports likelihood estimation, f-divergence estimation, mutual information estimation, and change point detection.

## Overview

Score-based generative modeling has emerged as a powerful paradigm for learning complex probability distributions. This framework unifies multiple approaches:

- **Score Matching Methods**: DRE∞, D3RE, OS-DRE, ISA-DRE
- **Kernel Methods**: Kernel Density Ratio Estimation
- **Neural Methods**: KULSIF, NCE, InfoNCE, RKL, Hellinger, KL, Gamma, EW, PW
- **Continuous Normalizing Flows**: ODE-based likelihood computation

### Key Features

- 🚀 **Unified Interface**: Single entry point (`main.py`) for all tasks and methods
- 📊 **Comprehensive Benchmarks**: Synthetic 2D, tabular, image datasets
- 🔧 **Flexible Configuration**: Extensive hyperparameter control via command-line args
- 📈 **Multi-Task Support**: Likelihood estimation, f-divergence, MI, CPD

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

## Architecture

```
OS_DRE/
├── main.py                    # Unified entry point
├── models/
│   ├── base_models.py         # Core neural network architectures
│   ├── SDE.py                 # SDE/ODE solvers (VPSDE, subVPSDE, Bridge)
│   ├── estimators.py          # Likelihood & density ratio estimators
│   ├── losses.py              # Training loss functions
│   ├── ncsn_unet.py           # Score-based UNet architecture
│   ├── layers.py              # Neural network layers
│   ├── pl_models.py           # PyTorch Lightning wrappers
│   ├── pl_data.py             # Data modules
│   └── nsf/                   # Neural Spline Flows integration
├── utils/
│   ├── config_setting.py       # Argument parsing & configuration
│   ├── data_utils.py          # Data generators
│   ├── sampling.py            # Sampling methods (HMC, AIS)
│   ├── visualization.py       # Plotting utilities
│   └── logger.py              # Logging utilities
└── results/                  # Experiment outputs
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
