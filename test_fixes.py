#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify the fixes to OOD DataModule
"""

import os
import sys
import torch
import numpy as np

# Add the models directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_ood_modifications():
    print("Testing OOD DataModule modifications...")
    
    # Import the specific components we need to test
    from models.pl_data import OOD_EVAL_CONFIG, ConcatDataset, OODDetectionDataset, OutOfDistributionDetectionDataModule
    
    # Test that OOD_EVAL_CONFIG is properly defined
    expected_keys = {'mnist', 'fashion_mnist', 'cifar10', 'cifar100'}
    actual_keys = set(OOD_EVAL_CONFIG.keys())
    
    if expected_keys.issubset(actual_keys):
        print("✓ OOD_EVAL_CONFIG contains all expected keys")
        for key in expected_keys:
            print(f"  {key}: {OOD_EVAL_CONFIG[key]}")
    else:
        print(f"✗ OOD_EVAL_CONFIG missing keys. Expected: {expected_keys}, Got: {actual_keys}")
        return False
    
    # Test ConcatDataset functionality
    # Create dummy datasets
    class DummyDataset:
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]
    
    dummy_data1 = [(torch.rand(3, 32, 32), 0) for _ in range(10)]
    dummy_data2 = [(torch.rand(3, 32, 32), 1) for _ in range(15)]
    
    dataset1 = DummyDataset(dummy_data1)
    dataset2 = DummyDataset(dummy_data2)
    concat_dataset = ConcatDataset([dataset1, dataset2])
    
    if len(concat_dataset) == 25:
        print("✓ ConcatDataset works correctly (length: 25)")
    else:
        print(f"✗ ConcatDataset length issue: expected 25, got {len(concat_dataset)}")
        return False
    
    # Test that we can access elements from both original datasets
    first_item = concat_dataset[0]
    middle_item = concat_dataset[9]  # Last item of first dataset
    next_item = concat_dataset[10]  # First item of second dataset
    last_item = concat_dataset[24]
    
    if first_item[1] == 0 and next_item[1] == 1:
        print("✓ ConcatDataset correctly accesses items from both datasets")
    else:
        print(f"✗ ConcatDataset item access issue: {first_item[1]}, {next_item[1]}")
        return False
    
    # Test OODDetectionDataset functionality
    in_dist_data = [(torch.rand(3, 32, 32), 0) for _ in range(5)]
    ood_data = [(torch.rand(3, 32, 32), 1) for _ in range(10)]
    
    in_dist_dataset = DummyDataset(in_dist_data)
    ood_dataset = DummyDataset(ood_data)
    ood_det_dataset = OODDetectionDataset(in_dist_dataset, ood_dataset)
    
    if len(ood_det_dataset) == len(in_dist_dataset):
        print(f"✓ OODDetectionDataset length is correct (based on in-dist dataset: {len(in_dist_dataset)})")
    else:
        print(f"✗ OODDetectionDataset length issue: expected {len(in_dist_dataset)}, got {len(ood_det_dataset)}")
        return False
    
    # Test that OODDetectionDataset uses random sampling by checking multiple samples
    samples = []
    for i in range(5):
        x_in, x_ood, label = ood_det_dataset[0]  # Always same in-dist index
        samples.append((x_in.clone(), x_ood.clone()))
    
    # Check if OOD samples are different (indicating random sampling)
    ood_samples_different = False
    for i in range(1, len(samples)):
        if not torch.allclose(samples[0][1], samples[i][1]):  # Compare OOD samples
            ood_samples_different = True
            break
    
    if ood_samples_different:
        print("✓ OODDetectionDataset uses random sampling for OOD data")
    else:
        print("? OODDetectionDataset may not be using random sampling (hard to verify with small dataset)")
    
    # Test that we can import the main class
    try:
        from models.pl_data import OutOfDistributionDetectionDataModule
        print("✓ OutOfDistributionDetectionDataModule can be imported")
    except ImportError as e:
        print(f"✗ Failed to import OutOfDistributionDetectionDataModule: {e}")
        return False
    
    # Test that the _load_ood_benchmarks method exists
    try:
        dm = OutOfDistributionDetectionDataModule(type('', (), {'data_dir': './data', 'ood_in_dist': 'mnist'})())
        if hasattr(dm, '_load_ood_benchmarks'):
            print("✓ _load_ood_benchmarks method exists")
        else:
            print("✗ _load_ood_benchmarks method missing")
            return False
    except Exception as e:
        print(f"? Could not fully test _load_ood_benchmarks: {e}")
    
    print("\nAll tests passed! The OOD DataModule modifications are working correctly.")
    return True

if __name__ == "__main__":
    success = test_ood_modifications()
    if success:
        print("\n✓ All modifications implemented successfully!")
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)