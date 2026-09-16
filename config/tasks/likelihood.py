# -*- coding: utf-8 -*-
"""Likelihood estimation task (`python main.py likelihood_estimation ...`)."""

from pl_data import REAL_IMAGE_DATASETS, TABULAR_DATASETS, SYNTHETIC2D_DATASETS

from ..common_args import get_common_args
from ..constants import DATASET_DIMS, SUB_METHODS, SUBSUB_CHOICES
from .image import set_image_args


def get_likelihood_args(parser):
    parser = get_common_args(parser)

    parser.add_argument(
        '--data', choices=[*SYNTHETIC2D_DATASETS, *TABULAR_DATASETS, *REAL_IMAGE_DATASETS],
        type=str, default='8gaussians')
    parser.add_argument("--method", type=str, default="dre", choices=["cnf", "dre"])
    parser.add_argument("--sub_method", type=str, default="score", choices=[*SUB_METHODS])
    parser.add_argument("--subsub_method", type=str, default="dre_infty", choices=SUBSUB_CHOICES)
    parser.add_argument("--subtask", type=str, default="synthetic2d", choices=["synthetic2d", "tabular", "real_image"])

    parser.add_argument('--batch_num', type=int, default=100)

    # real image datasets
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--layer_mode", type=str, default="conv")
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument(
        "--use_pretrained_flow",
        type=eval,
        default=False,
        choices=[True, False],
        help="Whether to use the pretrained NSF flow as a learnable data transform "
             "for real-image density estimation. If False (default), an identity "
             "transform is used and no NSF checkpoints are required.",
    )

    return parser


def _finalize_tabular(args):
    # python main.py likelihood_estimation --subtask tabular --data bsds300 --method dre --decomp --K_decomp 400 --batch_size 20000 --nhidden 5 --hdim_factor 1 --gpu_id 1
    args.d = DATASET_DIMS[args.data]

    args.grad_clip = 5.
    if args.decomp:
        args.hdim_factor = 2
        args.dims = '-'.join([str(64 * args.hdim_factor)] * args.nhidden)
    else:
        args.hdim_factor = 4
        hidden_dim = max(args.hdim_factor * args.d, 128)
        args.dims = '-'.join([str(hidden_dim)] * args.nhidden)
    return args


def _finalize_synthetic2d(args):
    # python main.py likelihood_estimation --subtask synthetic2d --data checkerboard --method dre --decomp --K_decomp 400
    args.specnorm = False
    args.norm = False
    if not args.test_only:
        args.quad_step = 20
    return args


def _finalize_real_image(args):
    # python main.py likelihood_estimation --subtask real_image --data mnist --method dre --decomp --K_decomp 1000 --gpu_id 0
    args = set_image_args(args)
    args.early_stopping = 30
    args.d = DATASET_DIMS[args.data]
    return args


_SUBTASK_FINALIZERS = {
    "tabular": _finalize_tabular,
    "synthetic2d": _finalize_synthetic2d,
    "real_image": _finalize_real_image,
}


def finalize_likelihood_args(args):
    """Per-subtask defaults applied after parsing."""
    finalizer = _SUBTASK_FINALIZERS.get(args.subtask)
    if finalizer is not None:
        args = finalizer(args)
    return args
