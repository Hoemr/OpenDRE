# -*- coding: utf-8 -*-
"""Method/solver registries and dataset dimensions shared by every task parser."""

from dre.estimators import SOLVER_DICT, QUAD_DICT, SOLVER_DICT_SCIPY

# ODE solvers and quadrature rules selectable via --solver
SOLVERS = [*SOLVER_DICT_SCIPY.keys(), *SOLVER_DICT.keys(), *QUAD_DICT.keys()]

# Density-ratio estimator families (--sub_method) and their variants (--subsub_method)
SUB_METHODS = ["score", "neural", "kernel"]
SUBSUB_METHODS = {"score": ["dre_infty", "d3re", "osdre", "isadre"],
                  "neural": ["kulsif", "logistic", "nce", "infonce", "rkl", "hellinger", "kl", "gamma", "ew", "pw", "chisq"],
                  "kernel": []}
SUBSUB_CHOICES = [subsub for sub_method in SUB_METHODS for subsub in SUBSUB_METHODS[sub_method]]

# Data dimensionality per dataset name (model input dim `d`)
DATASET_DIMS = {
    # 2D Toy Datasets
    'swissroll': 2, '8gaussians': 2, 'pinwheel': 2, 'circles': 2,
    'moons': 2, '2spirals': 2, 'checkerboard': 2, 'rings': 2,
    # Tabular Datasets
    'bsds300': 63, 'power': 6, 'gas': 8, 'hepmass': 21, 'miniboone': 43,
    # Image Datasets
    'mnist': 784, 'fashion_mnist': 784, 'cifar10': 3072, 'cifar100': 3072,
    'svhn': 3072, 'lsun': 3072, 'celeba': 3072, 'imagenet': 3072,
    # Change Point Detection Datasets
    'cpd_text': 20, 'cpd_creditcard': 4,
    # Mutual Information Datasets
    'diag_multi_gauss': 2, 'half_cube_map': 2,
}
