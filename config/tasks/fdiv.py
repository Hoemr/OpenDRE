# -*- coding: utf-8 -*-
"""f-divergence estimation task (`python main.py fdiv_estimation ...`).

The task hosts several subtasks; each one declares its own argument group
(`_add_*_args`) and its own post-parse defaults (`_finalize_*`).
"""

import pl_data

from ..common_args import get_common_args
from ..constants import DATASET_DIMS, SUB_METHODS, SUBSUB_CHOICES
from .image import set_image_args


def _add_task_args(parser):
    parser.add_argument(
        '--data', choices=[*pl_data.DISTRIBUTION_SHIFT_DATASETS, *pl_data.MI_DATASETS, *pl_data.MI_REAL_DATASETS, *pl_data.CHANGEPOINT_DATASETS],
        type=str, default='gauss_shift')
    parser.add_argument("--method", type=str, default="dre", choices=["dre", "MINE"])
    parser.add_argument("--sub_method", type=str, default="score", choices=[*SUB_METHODS])
    parser.add_argument("--subsub_method", type=str, default="dre_infty", choices=SUBSUB_CHOICES)
    parser.add_argument("--subtask", type=str, default="distribution_shift", choices=["distribution_shift", "mutual_information", "mi_realdata", "change_point_detection", "ood_detection"])

    # training parameters
    parser.add_argument('--batch_num', type=int, default=1)


def _add_distribution_shift_args(parser):
    parser.add_argument('--rho', type=float, default=0.5)
    parser.add_argument('--shift', type=float, default=0.05)
    parser.add_argument('--shift_steps', type=int, default=5)


def _add_mi_realdata_args(parser):
    # mutual_information for real data
    parser.add_argument("--data_dir", type=str, default="../data")
    # parser.add_argument("--mi_dtype", type=str, default="image", choices=["gaussian", "image", "text", "mixture"], help="Type of data to use for MI estimation")
    parser.add_argument("--mi_dname1", type=str, default="mnist", choices=["mnist", "cifar10", "cifar100"], help="Dataset name for image data")
    parser.add_argument("--mi_dname2", type=str, default="imdb.bert-imdb-finetuned", choices=["imdb.bert-imdb-finetuned", "imdb.roberta-imdb-finetuned"], help="Dataset name for text data")
    parser.add_argument("--mi_ds", type=int, default=10, help="Number of information sources")
    parser.add_argument("--mi_dr", type=int, default=10, help="Representation dimension")
    parser.add_argument("--mi_image_channels", type=int, default=1, help="Number of image channels (1 for grayscale, 3 for RGB)")
    parser.add_argument("--mi_image_patches", type=str, default="[1, 2, 5]", help="Layout of image patches as a list (e.g., [1,2,5] for 1x2x5 grid)")
    parser.add_argument("--mi_nuisance", type=float, default=0.0, help="Strength of background noise (0.0 to 1.0)")
    parser.add_argument("--mi_mode", type=str, default="stepwise", choices=["stepwise", "single"], help="Training mode: stepwise (varying MI) or single (fixed MI)")
    parser.add_argument("--mi_n_steps", type=int, default=20000, help="Total number of training steps")
    parser.add_argument("--mi_n_samples", type=int, default=50000, help="Number of samples")
    parser.add_argument("--true_mi", type=float, default=4.0, help="Target mutual information value in bits (for single mode)")


def _add_change_point_detection_args(parser):
    # change point detection
    parser.add_argument('--cpd_swl', type=int, default=64, help='Sliding window length for change point detection')
    parser.add_argument('--use_simplex', type=eval, default=False, choices=[True, False], help='Use simplex instead of Gaussian noise for CPD')


def _add_ood_detection_args(parser):
    # OOD detection
    parser.add_argument('--ood_in_dist', type=str, default='cifar10', choices=['mnist', 'fashion_mnist', 'cifar10', 'cifar100'], help='In-distribution dataset for OOD detection')
    parser.add_argument('--ood_use_imglist', type=eval, default=True, choices=[True, False], help='Use imglist-based OOD data under data_dir (see pl_data/ood_detection.py); only cifar10/cifar100 supported in imglist mode')
    parser.add_argument('--ood_base_type', type=str, default='universal', choices=['local', 'general', 'universal'], help='Base distribution type for OOD detection (built-in mode only)')
    parser.add_argument('--ood_crop_size', type=int, default=12, help='Crop size for local OOD detection')
    parser.add_argument('--ood_train', type=str, default="random", help='Override OOD training source for imglist mode: "selfgen" (q from ID transforms), "random", or path; default uses benchmark config')


def _add_image_backbone_args(parser):
    # Image backbone (ResNet or WideResNet for all image-related tasks)
    parser.add_argument('--image_backbone', type=str, default='unet', choices=['unet', 'resnet18', 'resnet34', 'resnet50', 'wideresnet28_10'], help='Backbone network for image-based tasks (OOD detection, real images, etc.)')
    parser.add_argument('--image_backbone_norm', type=str, default='groupnorm', choices=['batchnorm', 'groupnorm'], help='Norm layer in image backbone: batchnorm or groupnorm (groupnorm often more stable for small batches / OOD)')
    parser.add_argument('--image_backbone_num_groups', type=int, default=8, help='Number of groups for GroupNorm when image_backbone_norm=groupnorm')
    parser.add_argument('--image_dropout', type=float, default=0.3, help='Dropout rate for WideResNet backbone')


def get_fdiv_args(parser):
    parser = get_common_args(parser)

    _add_task_args(parser)
    _add_distribution_shift_args(parser)
    _add_mi_realdata_args(parser)
    _add_change_point_detection_args(parser)
    _add_ood_detection_args(parser)
    _add_image_backbone_args(parser)

    return parser


def _finalize_distribution_shift(args):
    # python main.py fdiv_estimation --method dre --subtask distribution_shift --data gauss_shift --decomp --K_decomp 300 --shift_steps 5 --shift 0.05 --d 80
    args.epochs = 10000
    # args.batch_size = 9600
    args.norm = False
    args.eval_freq = 100
    return args


def _finalize_change_point_detection(args):
    # python main.py fdiv_estimation --method dre --subtask change_point_detection --data cpd_text --use_simplex True --cpd_swl 64
    args.dims = "128-128-128"
    args.reload_interval = args.eval_freq * 5
    args.num_data_head = 1
    args.num_time_head = 1
    args.eval_freq = 100
    return args


def _finalize_mutual_information(args):
    # python main.py fdiv_estimation --method dre --subtask mutual_information --data diag_multi_gauss --decomp --K_decomp 1000 --epochs 40000 --d 160 --gpu_id 0
    # python main.py fdiv_estimation --method dre --subtask mutual_information --data half_cube_map --decomp --K_decomp 300 --epochs 10000 --d 2 --rho 0.5 --batch_size 4800 --gpu_id 0
    args.reload_interval = args.eval_freq * 5
    args.eval_freq = 100
    return args


def _finalize_mi_realdata(args):
    # base: python main.py fdiv_estimation --subtask mi_realdata
    # For image data with stepwise MI: --data image --mi_dname1 mnist --mi_ds 4 --mi_dr 4096 --mi_image_channels 1 --mi_image_patches "[2,2,1]" --mi_mode stepwise
    # For text data with fixed MI: --data text --mi_dname2 imdb.bert-imdb-finetuned --mi_ds 10 --mi_dr 7680 --mi_mode single --true_mi 8.0
    # For mixed data with background noise: --data mixture --mi_dname1 cifar10 --mi_dname2 imdb.bert-imdb-finetuned --mi_ds 6 --mi_dr 6144 --mi_image_channels 3 --mi_image_patches "[1,2,3]" --mi_nuisance 0.3 --mi_mode single --true_mi 6.0
    args = set_image_args(args)
    args.eval_freq = 1
    args.batch_num = args.mi_n_samples // args.batch_size
    args.epochs = args.mi_n_steps // args.batch_num
    try:
        image_patches = eval(args.mi_image_patches)
        if not isinstance(image_patches, list) or len(image_patches) != 3:
            raise ValueError("image_patches must be a list of three integers")
    except:
        raise ValueError("Invalid format for image_patches. Expected format: [x, y, z]")

    if args.data in ["image", "mixture"]:
        import numpy as np
        if np.prod(image_patches) != args.mi_ds:
            raise ValueError(f"Product of image_patches {image_patches} must equal ds={args.mi_ds}")
    return args


def _finalize_ood_detection(args):
    # python main.py fdiv_estimation --subtask ood_detection --ood_in_dist mnist --ood_base_type universal --epochs 1000 --batch_size 512 --lr 0.01
    # Set OOD detection specific args
    args = set_image_args(args)
    args.data = args.ood_in_dist
    args.d = DATASET_DIMS.get(args.data, DATASET_DIMS["mnist"])
    args.reload_interval = args.epochs
    return args


_SUBTASK_FINALIZERS = {
    "distribution_shift": _finalize_distribution_shift,
    "change_point_detection": _finalize_change_point_detection,
    "mutual_information": _finalize_mutual_information,
    "mi_realdata": _finalize_mi_realdata,
    "ood_detection": _finalize_ood_detection,
}


def finalize_fdiv_args(args):
    """Global defaults plus per-subtask defaults applied after parsing."""
    # args.num_data_head = 3
    # args.num_time_head = 3
    args.d = DATASET_DIMS.get(args.data, args.d)

    if args.d < 100:
        # args.dims = "128-128-128"
        args.dims = "256-256-256-256"
    elif args.d < 200:
        if args.decomp:
            args.dims = "512-512-512-512"
        args.dims = "256-256-256-256"
    else:
        args.dims = "512-512-512-512"

    args.reload_interval = args.epochs

    finalizer = _SUBTASK_FINALIZERS.get(args.subtask)
    if finalizer is not None:
        args = finalizer(args)
    return args
