# OpenDRE: A Unified Framework for Density Ratio Estimation

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository provides a unified implementation of density ratio estimation methods, integrating multiple state-of-the-art papers from top-tier machine learning venues (NeurIPS, ICML, ICLR). The framework supports likelihood estimation, f-divergence estimation, mutual information estimation, change point detection, and out-of-distribution (OOD) detection.

## Overview

Score-based generative modeling has emerged as a powerful paradigm for learning complex probability distributions. This framework unifies multiple approaches:

- **Score-based Methods**: DRE-$\infty$, D$^3$RE, OS-DRE, ISA-DRE, MVP-DRE
- **Kernel Methods**: Kernel Density Ratio Estimation
- **Neural Methods**: KULSIF, NCE, InfoNCE, RKL, Hellinger, KL, Gamma, EW, PW
- **Continuous Normalizing Flows**: ODE-based likelihood computation


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
| `nn_models/` | Neural network models (`tabular_models`, `image_models`, legacy `UnifiedModel` wrapper, plus logical `layers` modules) |
| `pl_data/` | Lightning DataModules and dataset configs |
| `pl_models/` | Lightning modules per task |
| `dre/` | Core DRE components: paths (`dre/path`), losses (`dre/losses`), estimators (`dre/estimators.py`), polynomial bases (`dre/polynomials.py`) |
| `core/` | Project helpers (utils.py, data_utils.py, tabular/), sampling routines, plotting/visualization, logger (`core/logger.py`), and helper scripts (e.g. `download_ood_benchmark.py`). Named `core` to avoid shadowing torchquad's `utils`. |
| `config/` | Argument parsing and save paths |

Import examples: `from pl_data import SUBTASK_DATA_MODULES`, `from nn_models import JointScoreModel`, `from config import parse_args`, `from path import ToyInterpXt`.

## Quick Start

### Command Line Interface

All experiments are run through `main.py` with subparsers for different tasks:

```bash
# Likelihood estimation
python main.py likelihood_estimation --subtask synthetic2d --data checkerboard --method dre --decomp --K_decomp 400

# Tabular density estimation
python main.py likelihood_estimation --subtask tabular --data bsds300 --method dre --decomp --K_decomp 400 --batch_size 20000

# Real image modeling
python main.py likelihood_estimation --subtask real_image --data mnist --method dre --decomp --K_decomp 400

# Distribution shift estimation
python main.py fdiv_estimation --subtask distribution_shift --data gauss_shift --decomp --K_decomp 300 --shift_steps 5 --shift 0.05 --d 80

# Mutual information estimation
python main.py fdiv_estimation --subtask mutual_information --data diag_multi_gauss --decomp --K_decomp 400 --epochs 40000 --d 160

# Change point detection
python main.py fdiv_estimation --subtask change_point_detection --data cpd_text --use_simplex True --cpd_swl 64

# OOD detection (see "OOD detection" section below for tasks and parameters)
python main.py fdiv_estimation --subtask ood_detection --sub_method neural --subsub_method nce --ood_use_imglist True --ood_in_dist cifar10 --image_backbone resnet18
python main.py fdiv_estimation --subtask ood_detection --sub_method score --subsub_method d3re --condition True --joint False --ood_use_imglist True --ood_in_dist fashion_mnist --image_backbone resnet18
```

### Unified CLI pattern (shared across tasks)

Most experiments follow the same high-level pattern:

```bash
python main.py <task> \
  --subtask <subtask> \
  --data <dataset> \
  --method <dre|cnf> \
  --sub_method <score|neural|kernel> \
  --subsub_method <method_variant> \
  [task-specific arguments...]
```

- **Common arguments**
  - `--task` (implied by subparser): `likelihood_estimation` or `fdiv_estimation`
  - `--subtask`: task-specific subtask (e.g. `synthetic2d`, `tabular`, `real_image`, `distribution_shift`, `mutual_information`, `change_point_detection`, `ood_detection`, `mi_realdata`)
  - `--data`: dataset name (depends on subtask; see tables below)
  - `--method`: high-level method family, currently `dre` (density ratio estimation) or `cnf` (continuous normalizing flows, for likelihood estimation)
  - `--sub_method`: estimator type:
    - `score`: score-based DRE (path-based models)
    - `neural`: neural density-ratio estimators (e.g. NCE, InfoNCE, logistic, PW, χ²)
    - `kernel`: kernel density-ratio estimators
  - `--subsub_method`: specific method variant within the chosen `sub_method`:
    - score-based: `dre_infty`, `d3re`, `osdre`, `isadre`
    - neural: `kulsif`, `logistic`, `nce`, `infonce`, `rkl`, `hellinger`, `kl`, `gamma`, `ew`, `pw`, `chisq`
    - kernel: currently no sub-variants

#### Per-task method configuration (examples)

- **Likelihood estimation (`task=likelihood_estimation`)**
  - 2D synthetic (score-based vs neural vs kernel):
    ```bash
    # Score-based DRE (e.g., DRE∞)
    python main.py likelihood_estimation \
      --subtask synthetic2d \
      --data checkerboard \
      --method dre \
      --sub_method score \
      --subsub_method dre_infty \
      --decomp --K_decomp 400

    # Neural DRE (e.g., NCE)
    python main.py likelihood_estimation \
      --subtask synthetic2d \
      --data checkerboard \
      --method dre \
      --sub_method neural \
      --subsub_method nce

    # Kernel DRE
    python main.py likelihood_estimation \
      --subtask synthetic2d \
      --data checkerboard \
      --method dre \
      --sub_method kernel
    ```
  - Tabular / real image tasks share the same pattern; only `--subtask`, `--data`, and some optimization hyperparameters (e.g., `--batch_size`, `--epochs`) change.

- **F-divergence estimation (`task=fdiv_estimation`)**
  - Distribution shift, mutual information, change point detection:
    ```bash
    # Score-based DRE for distribution shift
    python main.py fdiv_estimation \
      --subtask distribution_shift \
      --data gauss_shift \
      --method dre \
      --sub_method score \
      --subsub_method osdre \
      --decomp --K_decomp 300

    # Neural DRE for mutual information
    python main.py fdiv_estimation \
      --subtask mutual_information \
      --data diag_multi_gauss \
      --method dre \
      --sub_method neural \
      --subsub_method nce \
      --epochs 40000 --d 160
    ```

- **OOD detection (`task=fdiv_estimation`, `subtask=ood_detection`)**
  - The OOD pipeline is the most involved; the recommended combinations (score vs neural methods, CIFAR-10 vs CIFAR-100, train vs test, multi-seed runs) are encoded in `dre_ood.sh`.
  - Core patterns (per single run) are:
    ```bash
    # Neural OOD (e.g., NCE) with ResNet-18 backbone
    python main.py fdiv_estimation \
      --subtask ood_detection \
      --method dre \
      --sub_method neural \
      --subsub_method nce \
      --ood_use_imglist True \
      --ood_in_dist cifar10 \
      --image_backbone resnet18 \
      --epochs 1000 \
      --batch_size 768 \
      --lr 0.001 \
      --sample_noise_std 0.005 \
      --ood_train random

    # Score-based OOD (e.g., D3RE or OS-DRE) with UNet backbone
    python main.py fdiv_estimation \
      --subtask ood_detection \
      --method dre \
      --sub_method score \
      --subsub_method d3re \  # or dre_infty, osdre
      --ood_use_imglist True \
      --ood_in_dist cifar10 \
      --image_backbone unet \
      --path_type trigonometric \
      --t_mode lognorm \
      --condition True \
      --joint True \
      --bridge 1 \
      --epochs 1000 \
      --batch_size 768 \
      --lr 0.001 \
      --sample_noise_std 0.005 \
      --ood_train random
    ```
  - For large-scale, multi-GPU, multi-seed sweeps, use the provided launcher:
    ```bash
    # Test mode (single-GPU, timing-friendly)
    bash dre_ood.sh test score cifar10 "0"

    # Train all score-based methods on CIFAR-10 and CIFAR-100, multiple GPUs and seeds
    bash dre_ood.sh train score both "0 1 2 3" "1 2 3407"
    ```
    `dre_ood.sh` centralizes:
    - Shared OOD hyperparameters (epochs, LR, noise level, imglist mode, path type, etc.)
    - Method lists:
      - Neural: `NEURAL_METHODS=(nce pw chisq infonce logistic)`
      - Score: `SCORE_METHODS=(dre_infty d3re osdre)`
    - GPU scheduling logic for training mode.

## OOD detection: tasks and parameters

Data is read from the **data** folder (sibling to OS_DRE; path set by `--data_dir`, default `../data`). Two modes:

| Mode | Description | Parameters |
|------|-------------|------------|
| **Built-in** | ID data from torchvision (download to `data_dir`). OOD source and eval sets fixed per ID. | `--ood_in_dist`, `--ood_base_type`, `--image_backbone`, etc. |
| **Imglist** | ID and OOD from list files under `data_dir` (one line per `image_path label`). Benchmarks defined in `pl_data/ood_detection.py`. | `--ood_use_imglist True`, `--ood_in_dist` (cifar10 or cifar100), `--data_dir`, `--image_backbone`, etc. |

### Built-in OOD (default)

| Task | In-distribution | OOD source / eval | Command-line |
|------|-----------------|-------------------|--------------|
| MNIST OOD | MNIST | `--ood_base_type`: **local** (cropped MNIST), **general** (Fashion MNIST), **universal** (300K random images). Eval: Fashion, EMNIST. | `--ood_in_dist mnist --ood_base_type general --image_backbone resnet18` |
| Fashion MNIST OOD | Fashion MNIST | Same options. Eval: MNIST, EMNIST. | `--ood_in_dist fashion_mnist --ood_base_type general --image_backbone resnet18` |
| CIFAR-10 OOD | CIFAR-10 | Same options. Eval: SVHN, CIFAR-100, LSUN, CelebA. | `--ood_in_dist cifar10 --ood_base_type general --image_backbone wideresnet28_10` |
| CIFAR-100 OOD | CIFAR-100 | Same options. Eval: SVHN, CIFAR-10, LSUN, CelebA. | `--ood_in_dist cifar100 --ood_base_type general --image_backbone wideresnet28_10` |

- **Backbone**: `--image_backbone resnet18` (MNIST/Fashion); `--image_backbone wideresnet28_10` (CIFAR, recommended).
- **Optional**: `--sample_noise_std 0.1`, `--ood_crop_size 12` (local), `--image_dropout 0.3`.

### Imglist OOD (nearood / farood, OpenOOD layout)

Imglist mode uses **list files** (one line per `image_path label`) and matches the **OpenOOD** data layout. Use this when you want standard near-OOD / far-OOD splits (e.g. CIFAR-100, TinyImageNet, MNIST, SVHN as OOD).

**How to get the txt files and data**

1. **Option A (recommended):** From OS_DRE repo, run our script (uses same OpenOOD Google Drive IDs):
   ```bash
   pip install gdown
   python core/download_ood_benchmark.py --save_dir ../data
   ```
   This creates `../data/benchmark_imglist/` (all txt lists) and `../data/images_classic/{cifar10,cifar100,tin,mnist,svhn}/`. Then set `--data_dir ../data`.

2. **Option B:** Clone [OpenOOD](https://github.com/Jingkang50/OpenOOD) and run their download script so that `data/` contains:
   - `benchmark_imglist/` — all `.txt` list files (e.g. `cifar10/train_cifar10.txt`, `cifar10/test_cifar100.txt`, …)
   - `images_classic/` — image folders (cifar10, cifar100, tin, mnist, svhn, …)
   From OpenOOD repo (with `pip install gdown`):
   ```bash
   python scripts/download/download.py --contents datasets --datasets cifar-10 --save_dir /path/to/data --dataset_mode benchmark
   ```
   For CIFAR-10 + CIFAR-100 benchmarks you can use `--datasets ood_v1.5` and download both. The script fetches **benchmark_imglist** (txt) and the image datasets (cifar10, cifar100, tin, mnist, svhn, etc.) into `save_dir`.
3. Point OS_DRE at that folder: `--data_dir /path/to/data` (the directory that contains `benchmark_imglist` and `images_classic`).

**Near-OOD vs far-OOD (OpenOOD convention)**

- **Near-OOD**: semantically closer to ID (e.g. CIFAR-100, TinyImageNet when ID is CIFAR-10).
- **Far-OOD**: visually/semantically farther (e.g. MNIST, SVHN, Texture, Places365 when ID is CIFAR-10).

In `pl_data/ood_detection.py`, `OOD_IMGLIST_BENCHMARKS["cifar10"]["ood_eval"]` defines the evaluation OOD sets by name and list path; the same txt files are produced by OpenOOD’s download. You do not set nearood/farood via a single parameter — they are fixed by the benchmark (list of eval datasets). To add or remove OOD sets, edit `OOD_IMGLIST_BENCHMARKS` in `pl_data/ood_detection.py`.

**Layout after OpenOOD download**

```
data_dir/
  benchmark_imglist/
    cifar10/   train_cifar10.txt, val_cifar10.txt, test_cifar10.txt, val_tin.txt, test_cifar100.txt, test_tin.txt, test_mnist.txt, test_svhn.txt, ...
    cifar100/  ...
  images_classic/
    cifar10/, cifar100/, tin/, mnist/, svhn/, ...
```

**Example**

```bash
# After downloading with OpenOOD script and setting data_dir to that data folder
python main.py fdiv_estimation --subtask ood_detection --ood_use_imglist True --ood_in_dist cifar10 --image_backbone resnet18 --epochs 500  --sub_method neural --subsub_method nce
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

## Data Acquisition

This section summarizes where the data for each task/subtask comes from, where it is stored, and how to prepare it. The root data directory is specified by `--data_dir` (default `../data`).

### Likelihood Estimation

| Subtask | Source | Location / How to obtain |
|--------|--------|--------------------------|
| **synthetic2d** | Generated in code | No download required; datasets (e.g., swissroll, 8gaussians, pinwheel, etc.) are generated on the fly. |
| **tabular** | Local files (UCI, etc.) | Place data under `{data_dir}/tabular/` with the following structure:<br>• **BSDS300**: `BSDS300/BSDS300.hdf5`<br>• **power**: `power/data.npy`<br>• **gas**: `gas/ethylene_CO.pickle`<br>• **hepmass**: `hepmass/1000_train.csv`, `hepmass/1000_test.csv`<br>• **miniboone**: `miniboone/data.npy` |
| **real_image** | Torchvision download + local | **mnist / fashion_mnist / cifar10 / svhn / lsun_church / celeba**: downloaded via torchvision with `download=True` into `{data_dir}`.<br>**frey_face**: manually place `frey_rawface.mat` under `{data_dir}`. |

### $f$-Divergence Estimation

| Subtask | Source | Location / How to obtain |
|--------|--------|--------------------------|
| **distribution_shift** | Generated in code | Synthetic data (e.g., gauss_shift, beta_shift, cifar_c, domainbed); no external files needed. |
| **mutual_information** | Generated in code | Synthetic data (e.g., half_cube_map, gauss); no download required. |
| **mi_realdata** | Torchvision + local text data | Uses `args.data_dir`: image data (MNIST/CIFAR-10) is downloaded via torchvision; text data (IMDB, etc.) should be placed under the corresponding subdirectories of `{data_dir}` as expected by the code. |
| **change_point_detection** | Local files / synthetic | **cpd_mc**: synthetic, no files needed.<br>**cpd_text**: place `Language Detection.csv` under `./data/density_ratio_estimation/change_point_detection/` (or the path configured in the code).<br>**cpd_creditcard**: place `creditcard.txt` (or equivalent CSV) in the same directory. |
| **ood_detection** | See OOD section below | See [OOD data: how to prepare txt and images](#ood-data-txt-and-images). |

### OOD Data: txt and images

- **Built-in ID datasets (no imglist)**  
  ID datasets such as CIFAR-10/CIFAR-100 are downloaded automatically via torchvision into `{data_dir}`. An optional `300K_random_images.npy` can be used for universal OOD; you need to prepare this file yourself and place it at the expected path.

- **Imglist-style OOD (OpenOOD-style)**  
  Two options:  
  **Option A**: use the helper script  
  `python core/download_ood_benchmark.py`  
  This downloads `benchmark_imglist` and `images_classic` into the configured data directory (either from the script or environment variables).  
  **Option B**: clone [OpenOOD](https://github.com/Jingkang50/OpenOOD) and prepare `benchmark_imglist/` and `images_classic/` as described in its documentation, then place them under `{data_dir}` so that the layout matches `pl_data/ood_detection.py` (`OOD_IMGLIST_BENCHMARKS`, `list_subdir`/`image_subdir`, including near- and far-OOD).  
  When using imglist mode, set `--ood_use_imglist True`; only `cifar10` and `cifar100` are supported as ID datasets.

## Method Configuration

### Score Estimation Methods (`--sub_method score`)

| Path Type | Description | Flag |
|-----------------|-------------|------|
| `VP` | Variance Preserving SDE | Default |
| `subVP` | Sub-Variance Preserving SDE | `--path_type subVP` |
| `bridge` | Bridge Matching | `--bridge` |
| `mvp` | Mimimum Variance Path | `--mvp` |
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
| `--sample_noise_std` | 0.1 | Noise standard deviation (unified name) |
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

Models live in `nn_models/`:

- `tabular_models.py`: tabular score and density-ratio models
- `image_models.py`: image-based UNet/ResNet/WideResNet score and density-ratio models
- `unified.py`: thin **compatibility wrapper** (`UnifiedModel`) that dispatches to the
  concrete models above. New code should use `tabular_models` / `image_models` directly.

Use the existing `pl_models`’ `build_model()` methods as the reference for constructor
arguments and how models are wired into tasks.

### Existing model classes

| Use case | Tabular | Image (UNet / ResNet / WRN) |
|----------|---------|-----------------------------|
| Score \(s(x, t)\) | `JointScoreModel`, `DecompJointScoreModel` | `ImageJointScoreModel` (UNet), `ImageJointScoreModelResNet`, `ImageDecompJointScoreModel` (UNet), `ImageDecompJointScoreModelResNet` |
| Density ratio \(r(x) = p(x)/q(x)\) | `NeuralDensityRatioModel` | `ResNetDensityRatioModel` |

Example (tabular joint score model):

```python
from nn_models import JointScoreModel

model = JointScoreModel(
    input_dim=args.d,
    hidden_dims=[256, 256, 256],
    embed_dim=args.embed_dim,
    nonlinearity=args.nonlinearity,
    args=args,
)
```

For image backbones (UNet / ResNet / WideResNet), see `pl_models/ood_detection.py` or
`pl_models/density_estimation_real_image.py` for how the Lightning modules build the
corresponding nn_models based on `args.image_backbone`, `args.image_backbone_norm`,
`args.image_dropout`, etc. Activation and polynomial options are documented in
`layers/activations.py` and in the model docstrings in `tabular_models.py` /
`image_models.py`.

### Adding a new model

You **do not** need to modify `UnifiedModel` when adding a new model. Instead:

- **New tabular model**
  1. Implement a new class in `nn_models/tabular_models.py` (e.g. `MyTabularJointScoreModel`),
     following the patterns of `JointScoreModel` or `DecompJointScoreModel`.
  2. If a task should use this model, update the corresponding `pl_models/*` Lightning
     module (`build_model()` or `_build_default_model()`) to instantiate your class.

- **New image backbone or image model**
  1. Implement a new class in `nn_models/image_models.py` (e.g. `MyImageJointScoreModel`),
     or extend the ResNet/UNet helpers if you add a new backbone.
  2. Wire it in `pl_models` where image models are built, typically in
     `pl_models/ood_detection.py`, `pl_models/density_estimation_real_image.py`, or
     `pl_models/mi_real.py`, based on `args.image_backbone` or a new flag you add.

- **Optional compatibility layer**
  - Only if you still rely on `UnifiedModel` in legacy scripts, you can extend
    `nn_models/unified.py` to route a new `(data_type, model_type, backbone)` combination
    to your new concrete model. For all new code paths, prefer using
    `tabular_models` / `image_models` directly.

## Directory Layout

```
OpenDRE/
├── main.py
├── config/           # Argument parsing (e.g. config_setting.py)
├── pl_data/          # DataModules and dataset configs
├── pl_models/        # Lightning modules per task
├── nn_models/        # Model architectures (tabular_models, image_models, legacy UnifiedModel wrapper, plus layers)
├── nsf/              # Original NSF codebase (flows, datasets, experiments, checkpoints)
├── dre/              # Core DRE components: paths, losses, estimators, polynomial bases
├── core/             # Utils, data_utils, logger, sampling, visualization, helper scripts
└── results/          # Experiment logs, figures, and tables
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
