#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple import test for the modified OOD DataModule
"""

from models.pl_data import OutOfDistributionDetectionDataModule

class Args:
    data_dir = './data'
    seed = 42
    data = 'test'
    batch_size = 32
    test_batch_size = 32
    ood_in_dist = 'mnist'
    ood_base_type = 'universal'
    ood_noise_std = 0.1
    ood_anneal_alpha_init = 0.0
    ood_crop_size = 12
    gpu_id = [0]

print("Import successful")

# Create an instance to test basic functionality
try:
    args = Args()
    dm = OutOfDistributionDetectionDataModule(args)
    print("Instance creation successful")
except Exception as e:
    print(f"Instance creation failed: {e}")