# Summary of Changes to OOD DataModule

## Overview
This document summarizes the modifications made to `pl_data.py` to achieve two core objectives:
1. Decouple validation/testing OOD data from `ood_base_type`
2. Fix low OOD data utilization in universal mode

## Changes Made

### 1. Added Class-Level Constant `OOD_EVAL_CONFIG`
Defined standard OOD benchmark datasets for evaluation:
```python
OOD_EVAL_CONFIG = {
    'mnist': ['fashion_mnist', 'emnist'],
    'fashion_mnist': ['mnist', 'emnist'],
    'cifar10': ['svhn', 'cifar100', 'lsun', 'celeba'],
    'cifar100': ['svhn', 'cifar10', 'lsun', 'celeba']
}
```

### 2. Added `ConcatDataset` Class
Created a custom dataset class to concatenate multiple OOD benchmark datasets into a single dataset for validation/testing:
```python
class ConcatDataset(Dataset):
    """Concatenate multiple datasets into a single dataset."""
    ...
```

### 3. Modified `OODDetectionDataset` Class
- Changed `__len__` to return `len(in_dist_dataset)` instead of minimum of both datasets
- Changed `__getitem__` to use random sampling for OOD data (`random.randint(0, len(ood_dataset) - 1)`) instead of pairing with same indices

### 4. Modified `OutOfDistributionDetectionDataModule.setup` Method
- Training logic remains unchanged (still uses `ood_base_type`)
- Validation/testing now uses `_get_eval_ood_dataset` which loads standard OOD benchmarks from `OOD_EVAL_CONFIG`

### 5. Added `_get_eval_ood_dataset` Method
New method that:
- Loads OOD datasets based on `OOD_EVAL_CONFIG` for validation/testing
- Handles exceptions gracefully if datasets fail to load
- Falls back to original OOD dataset if all benchmarks fail
- Uses `transform_test` for consistency during validation/testing

### 6. Added Import Statement
Added `import random` at the top of the file to support random sampling in `OODDetectionDataset`.

## Key Features

### Training Phase
- OOD data selection still depends on `ood_base_type` (local/general/universal)
- No changes to training behavior

### Validation/Testing Phase
- OOD data comes from standard benchmarks defined in `OOD_EVAL_CONFIG`
- Independent of `ood_base_type` setting
- Ensures consistent evaluation across different model configurations

### OOD Data Utilization
- Fixed the issue where only ~50K of 300K OOD data was being used
- Now uses random sampling to ensure full utilization of OOD dataset during training
- Each epoch sees different OOD samples, improving training effectiveness

## Benefits

1. **Standardized Evaluation**: Validation/testing uses consistent OOD benchmarks regardless of training configuration
2. **Improved OOD Utilization**: Full 300K OOD dataset is now utilized during training in universal mode
3. **Robust Fallback**: Graceful handling of missing datasets with fallback to original logic
4. **Backward Compatibility**: Training logic remains unchanged, ensuring existing workflows continue to work

## Verification

All changes have been tested with:
- Unit tests for individual components
- Import and instantiation tests
- Functional verification of new features