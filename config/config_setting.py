# -*- coding: utf-8 -*-
import argparse
import torch

from nn_models.layers.activations import NONLINEARITIES
import pl_data
from pl_data import REAL_IMAGE_DATASETS, TABULAR_DATASETS, SYNTHETIC2D_DATASETS
from dre.estimators import SOLVER_DICT, QUAD_DICT, SOLVER_DICT_SCIPY

SOLVERS = [*SOLVER_DICT_SCIPY.keys(), *SOLVER_DICT.keys(), *QUAD_DICT.keys()]
SUB_METHODS = ["score", "neural", "kernel"]
SUBSUB_METHODS = {"score": ["dre_infty", "d3re"],
                  "neural": ["kulsif", "logistic", "nce", "infonce", "rkl", "hellinger", "kl", "gamma", "ew", "pw", "chisq"],
                  "kernel": []}

def get_common_args(parser):
    parser.add_argument(
        "--layer_type", type=str, default="concatsquashextend",
        choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatsquashextend", "concatcoord", "hyper", "blend"])
    parser.add_argument('--d', type=int, default=2)
    
    # network parameters
    parser.add_argument('--dims', type=str, default='128-128-128-128-128')
    parser.add_argument('--nhidden', type=int, default=5)
    parser.add_argument('--hdim_factor', type=int, default=5)
    parser.add_argument("--time_act", type=str, default="identity")
    parser.add_argument("--nonlinearity", type=str, default="leakyrelu", choices=NONLINEARITIES)
    parser.add_argument('--num_time_head', type=int, default=1)  # 3
    parser.add_argument('--num_data_head', type=int, default=1)  # 2
    parser.add_argument("--norm", type=eval, default=True, choices=[True, False])
    parser.add_argument("--specnorm", type=eval, default=False, choices=[True, False])
    parser.add_argument("--pre_norm", type=eval, default=False, choices=[True, False])
    parser.add_argument("--embed_dim", type=int, default=512)
    
    # training parameters
    parser.add_argument('--batch_size', type=int, default=9600)
    parser.add_argument('--test_batch_size', type=int, default=5000)
    parser.add_argument('--epochs', type=int, default=3000)   
    parser.add_argument('--eval_freq', type=int, default=10) 
    parser.add_argument('--gpu_id', type=int, nargs='+', default=[0])
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--grad_clip', type=float, default=1.)
    parser.add_argument('--accumu', type=int, default=1)   # accumulation step
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--early_stopping', type=int, default=30)
    
    # model parameters
    parser.add_argument("--eps", type=float, default=1e-5, help="t \in [eps, 1-eps]")
    parser.add_argument("--joint", type=eval, default=True, choices=[True, False])   
    parser.add_argument("--condition", type=eval, default=False, choices=[True, False]) 
    parser.add_argument('--path_type', type=str, default="VP", help="interpolant type")  
    parser.add_argument('--relative_ratio', type=float, default=0.2, help="relative ratio for relative ratio sampling")
    parser.add_argument('--scale', type=float, default=1., help="scale of loss terms")
    
    # solver parameters
    parser.add_argument('--solver', type=str, default='trapz', choices=SOLVERS)
    parser.add_argument('--quad_step', type=int, default=100)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-5)
    
    # bridge model
    parser.add_argument("--bridge", action='store_true', default=False)
    parser.add_argument("--OT", action='store_true', default=False)
    parser.add_argument('--sample_noise_std', type=float, default=0.005, help='Std of additive Gaussian noise for sample perturbation')
    parser.add_argument('--gamma_t', type=float, default=0.01)

    # EDM-style input preconditioning (path-aware total scale)
    parser.add_argument("--use_edm_precond", action='store_true', default=False)
    parser.add_argument("--edm_sigma_data", type=float, default=1.0)
    parser.add_argument("--edm_sigma_min", type=float, default=1e-3)
    parser.add_argument("--edm_use_noise_embed", action='store_true', default=False,
                        help="If set, add noise_embed(c_noise) to time_emb; default False (t-only preconditioning).")
    parser.add_argument("--edm_debug", action='store_true', default=False,
                        help="If set, log EDM preconditioning stats (s_t, x_t, x_t^in) for verification.")
    parser.add_argument("--edm_data_var_mode", type=str, default="channelwise", choices=["scalar", "channelwise"])
    parser.add_argument("--edm_stat_type", type=str, default="second_moment", choices=["variance", "second_moment"],
                        help="Endpoint stat: second_moment E[x^2] (default, RMS) or variance E[(x-mu)^2]")

    # Minimum Variance Path
    parser.add_argument("--mvp", action='store_true', default=False, help="Minimum Variance Path / Gaussian Mixture / Kumaraswamy Mixture")
    parser.add_argument('--grid_size', type=int, default=200)
    parser.add_argument('--K_gmm', type=int, default=5, help="num of components of GMM or KMM")
    parser.add_argument('--constraint_type', type=str, default="spherical", choices=["spherical", "affine"])  
  
    # decomposition (removed — OS-DRE only)

    # Engery modeling
    parser.add_argument("--energy", action='store_true', default=False)

    # Secant Alignment Identity
    parser.add_argument("--SAI", action='store_true', default=False)
    parser.add_argument("--t_mode", type=str, default="uniform", choices=["uniform", "IS", "lognorm"])
    parser.add_argument("--aligned_ratio", type=float, default=0.5)

    # Auxiliary Bregman loss for score matching
    parser.add_argument("--auxiliary_loss", action='store_true', default=False)
    parser.add_argument("--auxiliary_quad_steps", type=int, default=5)
    parser.add_argument("--bregman_weight", type=float, default=1.0)
    parser.add_argument(
        "--bregman_type",
        type=str,
        default="logistic",
        choices=["logistic", "kl", "pearson", "itakura_saito"],
    )
    parser.add_argument("--clip_tau", type=float, default=5.0)

    # sampling
    parser.add_argument("--sampling_only", action='store_true', default=False)
    parser.add_argument("--test_only", action='store_true', default=False)
    return parser

def get_likelihood_args(parser):
    parser = get_common_args(parser)

    parser.add_argument(
        '--data', choices=[*SYNTHETIC2D_DATASETS, *TABULAR_DATASETS, *REAL_IMAGE_DATASETS],
        type=str, default='8gaussians')
    parser.add_argument("--method", type=str, default="dre", choices=["cnf", "dre"])
    parser.add_argument("--sub_method", type=str, default="score", choices=[*SUB_METHODS])
    parser.add_argument("--subsub_method", type=str, default="dre_infty", choices=[subsub for sub_method in SUB_METHODS for subsub in SUBSUB_METHODS[sub_method]])
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

def get_fdiv_args(parser):
    parser = get_common_args(parser)

    parser.add_argument(
        '--data', choices=[*pl_data.DISTRIBUTION_SHIFT_DATASETS, *pl_data.MI_DATASETS, *pl_data.MI_REAL_DATASETS, *pl_data.CHANGEPOINT_DATASETS],
        type=str, default='gauss_shift')
    parser.add_argument("--method", type=str, default="dre", choices=["dre", "MINE"])
    parser.add_argument("--sub_method", type=str, default="score", choices=[*SUB_METHODS])
    parser.add_argument("--subsub_method", type=str, default="dre_infty", choices=[subsub for sub_method in SUB_METHODS for subsub in SUBSUB_METHODS[sub_method]])
    parser.add_argument("--subtask", type=str, default="distribution_shift", choices=["distribution_shift", "mutual_information", "mi_realdata", "change_point_detection", "ood_detection"])
    
    # training parameters
    parser.add_argument('--batch_num', type=int, default=1)
    
    # distribution_shift
    parser.add_argument('--rho', type=float, default=0.5)
    parser.add_argument('--shift', type=float, default=0.05)
    parser.add_argument('--shift_steps', type=int, default=5)
    
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
    
    # change point detection
    parser.add_argument('--cpd_swl', type=int, default=64, help='Sliding window length for change point detection')
    parser.add_argument('--use_simplex', type=eval, default=False, choices=[True, False], help='Use simplex instead of Gaussian noise for CPD')
    
    # OOD detection
    parser.add_argument('--ood_in_dist', type=str, default='cifar10', choices=['mnist', 'fashion_mnist', 'cifar10', 'cifar100'], help='In-distribution dataset for OOD detection')
    parser.add_argument('--ood_use_imglist', type=eval, default=True, choices=[True, False], help='Use imglist-based OOD data under data_dir (see pl_data/ood_detection.py); only cifar10/cifar100 supported in imglist mode')
    parser.add_argument('--ood_base_type', type=str, default='universal', choices=['local', 'general', 'universal'], help='Base distribution type for OOD detection (built-in mode only)')
    parser.add_argument('--ood_crop_size', type=int, default=12, help='Crop size for local OOD detection')
    parser.add_argument('--ood_train', type=str, default="random", help='Override OOD training source for imglist mode: "selfgen" (q from ID transforms), "random", or path; default uses benchmark config')
 
    # Image backbone (ResNet or WideResNet for all image-related tasks)
    parser.add_argument('--image_backbone', type=str, default='unet', choices=['unet', 'resnet18', 'resnet34', 'resnet50', 'wideresnet28_10'], help='Backbone network for image-based tasks (OOD detection, real images, etc.)')
    parser.add_argument('--image_backbone_norm', type=str, default='groupnorm', choices=['batchnorm', 'groupnorm'], help='Norm layer in image backbone: batchnorm or groupnorm (groupnorm often more stable for small batches / OOD)')
    parser.add_argument('--image_backbone_num_groups', type=int, default=8, help='Number of groups for GroupNorm when image_backbone_norm=groupnorm')
    parser.add_argument('--image_dropout', type=float, default=0.3, help='Dropout rate for WideResNet backbone')
    return parser

def set_image_args(args):
    # args.sample_noise_std = 0.
    # args.eval_freq = 10
    args.embed_dim = 128
    args.layer_type = "conv"    # useless now
    args.dims = "128-256-512"  # 128-256-512   or 64-128-256
    args.grad_clip = 10.
    # if args.data in ["mnist", "image", "fashion_mnist"]:
    #     args.batch_size = 1024
    #     args.test_batch_size = 1024
    # else:
    #     args.batch_size = 512
    #     args.test_batch_size = 512
    return args

def parse_args():
    parser = argparse.ArgumentParser(description="ODE-based models")
    subparsers = parser.add_subparsers(dest="task", required=True)   # subparser class
    
    # Subparser for likelihood estimation
    likelihood_parser = subparsers.add_parser("likelihood_estimation", conflict_handler='resolve')  # subparser instance
    likelihood_parser = get_likelihood_args(likelihood_parser)
    
    # Subparser for f divergence estimation
    fdiv_parser = subparsers.add_parser("fdiv_estimation")  # subparser instance
    fdiv_parser = get_fdiv_args(fdiv_parser)

    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.gpu_id[0]}" if torch.cuda.is_available() else "cpu")
    args.test_only = True if args.sampling_only else args.test_only
    args.dropout = 0 if args.condition else args.dropout
    
    # MinPV
    args.path_type = "mvp" if args.mvp else args.path_type
    args.mvp = True if args.path_type == "mvp" else args.mvp

    if args.subsub_method == "dre_infty":
        args.sub_method = "score"
        args.bridge = False
    elif args.subsub_method == "d3re":
        args.sub_method = "score"
        args.bridge = True

    # Energy modeling
    args.joint = False if args.energy else args.joint

    ds_dict = {
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

    # --condition True --t_mode IS
    if args.task == "likelihood_estimation":
        if args.subtask == "tabular":
            # python main.py likelihood_estimation --subtask tabular --data bsds300 --method dre --batch_size 20000 --nhidden 5 --hdim_factor 1 --gpu_id 1
            args.d = ds_dict[args.data]

            args.grad_clip = 5.
            args.hdim_factor = 4
            hidden_dim = max(args.hdim_factor * args.d, 128)
            args.dims = '-'.join([str(hidden_dim)] * args.nhidden)
        
        elif args.subtask == "synthetic2d":
            # python main.py likelihood_estimation --subtask synthetic2d --data checkerboard --method dre
            args.specnorm = False
            args.norm = False
            if not args.test_only:
                args.quad_step = 20
        
        elif args.subtask == "real_image":
            # python main.py likelihood_estimation --subtask real_image --data mnist --method dre --gpu_id 0
            args = set_image_args(args)
            args.early_stopping = 30 
            args.d = ds_dict[args.data]
    
    elif args.task == "fdiv_estimation":
        # args.num_data_head = 3
        # args.num_time_head = 3
        args.d = ds_dict.get(args.data, args.d)
        
        if args.d < 100:
            args.dims = "256-256-256-256"
        elif args.d < 200:
            args.dims = "256-256-256-256"
        else:
            args.dims = "512-512-512-512"

        args.reload_interval = args.epochs
        if args.subtask == "distribution_shift":
            # python main.py fdiv_estimation --method dre --subtask distribution_shift --data gauss_shift --shift_steps 5 --shift 0.05 --d 80
            args.epochs = 10000
            # args.batch_size = 9600
            args.norm = False
            args.eval_freq = 100
            
        elif args.subtask == "change_point_detection":
            # python main.py fdiv_estimation --method dre --subtask change_point_detection --data cpd_text --use_simplex True --cpd_swl 64
            args.dims = "128-128-128"
            args.reload_interval = args.eval_freq * 5
            args.num_data_head = 1
            args.num_time_head = 1
            args.eval_freq = 100
        
        elif args.subtask == "mutual_information":
            # python main.py fdiv_estimation --method dre --subtask mutual_information --data diag_multi_gauss --epochs 40000 --d 160 --gpu_id 0
            # python main.py fdiv_estimation --method dre --subtask mutual_information --data half_cube_map --epochs 10000 --d 2 --rho 0.5 --batch_size 4800 --gpu_id 0
            args.reload_interval = args.eval_freq * 5
            args.eval_freq = 100
        
        elif args.subtask == "mi_realdata":
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
        
        elif args.subtask == "ood_detection":
            # python main.py fdiv_estimation --subtask ood_detection --ood_in_dist mnist --ood_base_type universal --epochs 1000 --batch_size 512 --lr 0.01
            # Set OOD detection specific args
            args = set_image_args(args)
            args.data = args.ood_in_dist
            args.d = ds_dict.get(args.data, ds_dict["mnist"])
            args.reload_interval = args.epochs
    
    return args

def get_save_path(args):
    loss_type = "joint" if args.joint else "time"
    key_args = [args.method]
    key_args.extend([args.sub_method, args.subsub_method])
    if args.sub_method == "score":
        key_args.extend([args.path_type])
    if args.task == "likelihood_estimation":
        pass
        
    elif args.task == "fdiv_estimation":
        key_args.extend([f"d{args.d}"])
            
        if args.subtask == "mutual_information":
            key_args.extend(["rho"+str(args.rho)])
        
        elif args.subtask == "ood_detection":
            key_args.extend([args.ood_in_dist, args.image_backbone, args.ood_train, f"noise{args.sample_noise_std}",])
            if args.use_edm_precond:
                key_args.extend(["edm"])
            if getattr(args, "ood_use_imglist", False):
                key_args.append("imglist")
  
    else:
        raise ValueError(f"Task {args.task} is not implemented!")
    
    if args.sub_method == "score":
        if args.bridge:
            key_args.extend(["bridge" + str(args.gamma_t)])
        
        key_args.extend([args.t_mode])
    
        if args.SAI:
            key_args.extend([f"ratio{args.aligned_ratio}"])
    
        if args.condition:
            key_args.extend(["cond"])
        if args.OT:
            key_args.extend(["OT"])
        if args.path_type == "mvp":
            key_args.extend([args.constraint_type, f"K_{args.K_gmm}"])
        # Bregman auxiliary loss weight (for ablation)
        # bw = getattr(args, "bregman_weight", 1.0)
        # key_args.extend([f"bregman_w{bw}"])
        key_args.extend([loss_type])
    key_args.extend([f"seed{args.seed}"])
    
    save_path = f'./results/{args.task}/{args.subtask}/' + f'{args.data}/' + '_'.join(filter(None, key_args)) 
    return save_path
