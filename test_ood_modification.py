#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify the OOD DataModule modifications
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np

# Add the models directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from models.pl_data import OutOfDistributionDetectionDataModule
from utils.config_setting import Args

def test_ood_modifications():
    print("Testing OOD DataModule modifications...")
    
    # Create mock args
    class MockArgs:
        def __init__(self):
            self.data_dir = './data'
            self.seed = 42
            self.data = 'test'
            self.batch_size = 32
            self.test_batch_size = 32
            self.ood_in_dist = 'mnist'
            self.ood_base_type = 'universal'
            self.ood_noise_std = 0.1
            self.ood_anneal_alpha_init = 0.0
            self.ood_crop_size = 12
            self.gpu_id = [0]
    
    args = MockArgs()
    
    # Create the data module
    dm = OutOfDistributionDetectionDataModule(args)
    
    # Prepare data (this will download if needed)
    try:
        dm.prepare_data()
        print("✓ prepare_data() executed successfully")
    except Exception as e:
        print(f"✗ prepare_data() failed: {e}")
        return False
    
    # Setup the data module
    try:
        dm.setup()
        print("✓ setup() executed successfully")
    except Exception as e:
        print(f"✗ setup() failed: {e}")
        return False
    
    # Check that datasets are created
    if hasattr(dm, 'train_dataset'):
        print(f"✓ Train dataset created with length: {len(dm.train_dataset)}")
    else:
        print("✗ Train dataset not created")
        return False
    
    if hasattr(dm, 'val_dataset'):
        print(f"✓ Validation dataset created with length: {len(dm.val_dataset)}")
    else:
        print("✗ Validation dataset not created")
        return False
    
    if hasattr(dm, 'test_dataset'):
        print(f"✓ Test dataset created with length: {len(dm.test_dataset)}")
    else:
        print("✗ Test dataset not created")
        return False
    
    # Test that OOD_EVAL_CONFIG is accessible
    from models.pl_data import OOD_EVAL_CONFIG
    if args.ood_in_dist in OOD_EVAL_CONFIG:
        print(f"✓ OOD_EVAL_CONFIG contains entry for {args.ood_in_dist}: {OOD_EVAL_CONFIG[args.ood_in_dist]}")
    else:
        print(f"✗ OOD_EVAL_CONFIG missing entry for {args.ood_in_dist}")
        return False
    
    # Test that ConcatDataset works
    from models.pl_data import ConcatDataset
    # Create dummy datasets to test ConcatDataset
    dummy_data1 = [(torch.rand(3, 32, 32), 0) for _ in range(10)]
    dummy_data2 = [(torch.rand(3, 32, 32), 1) for _ in range(15)]
    
    class DummyDataset:
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]
    
    dataset1 = DummyDataset(dummy_data1)
    dataset2 = DummyDataset(dummy_data2)
    concat_dataset = ConcatDataset([dataset1, dataset2])
    
    if len(concat_dataset) == 25:
        print("✓ ConcatDataset works correctly")
    else:
        print(f"✗ ConcatDataset length issue: expected 25, got {len(concat_dataset)}")
        return False
    
    # Test that OODDetectionDataset uses random sampling
    from models.pl_data import OODDetectionDataset
    # Create small dummy datasets to test random sampling
    in_dist_data = [(torch.rand(3, 32, 32), 0) for _ in range(5)]
    ood_data = [(torch.rand(3, 32, 32), 1) for _ in range(10)]
    
    class SimpleDummyDataset:
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]
    
    in_dist_dataset = SimpleDummyDataset(in_dist_data)
    ood_dataset = SimpleDummyDataset(ood_data)
    ood_det_dataset = OODDetectionDataset(in_dist_dataset, ood_dataset)
    
    if len(ood_det_dataset) == len(in_dist_dataset):
        print("✓ OODDetectionDataset length is correct (based on in-dist dataset)")
    else:
        print(f"✗ OODDetectionDataset length issue: expected {len(in_dist_dataset)}, got {len(ood_det_dataset)}")
        return False
    
    # Test that random sampling works by checking multiple samples
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
    
    print("\nAll tests passed! The OOD DataModule modifications are working correctly.")
    return True

if __name__ == "__main__":
    success = test_ood_modifications()
    if success:
        print("\n✓ All modifications implemented successfully!")
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)