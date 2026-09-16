# -*- coding: utf-8 -*-
"""Argument parsing for every task.

Layout:
    constants.py     registries shared by all tasks (solvers, methods, dataset dims)
    common_args.py   arguments common to all tasks + global post-parse defaults
    tasks/           per-task arguments and post-parse defaults
    parser.py        `parse_args()` entry point
    save_path.py     `get_save_path()` result-directory naming
"""

from .constants import DATASET_DIMS, SOLVERS, SUBSUB_CHOICES, SUBSUB_METHODS, SUB_METHODS
from .common_args import finalize_common_args, get_common_args
from .tasks import finalize_fdiv_args, finalize_likelihood_args, get_fdiv_args, get_likelihood_args, set_image_args
from .parser import parse_args
from .save_path import get_save_path

__all__ = [
    "parse_args",
    "get_save_path",
    "get_common_args",
    "finalize_common_args",
    "get_likelihood_args",
    "finalize_likelihood_args",
    "get_fdiv_args",
    "finalize_fdiv_args",
    "set_image_args",
    "SOLVERS",
    "SUB_METHODS",
    "SUBSUB_METHODS",
    "SUBSUB_CHOICES",
    "DATASET_DIMS",
]
