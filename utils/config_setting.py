# -*- coding: utf-8 -*-
import argparse
import torch

from models import base_models
from models import polynomials as Pk

import models.pl_data as pl_data
from models.pl_data import REAL_IMAGE_DATASETS, TABULAR_DATASETS, SYNTHETIC2D_DATASETS
from models.estimators import SOLVER_DICT, QUAD_DICT, SOLVER_DICT_SCIPY

SOLVERS = [*SOLVER_DICT_SCIPY.keys(), *SOLVER_DICT.keys(), *QUAD_DICT.keys()]
SUB_METHODS = ["score", "neural", "kernel"]
SUBSUB_METHODS = {"score": ["dre_infty", "d3re", "os_dre", "isa_dre"], 
                  "neural": ["kulsif", "logistic", "nce", "infonce", "rkl", "hellinger", "kl", "gamma", "ew", "pw"], 
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
    parser.add_argument("--nonlinearity", type=str, default="leakyrelu", choices=base_models.NONLINEARITIES)
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
    parser.add_argument('--eval_freq', type=int, default=50) 
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
    parser.add_argument('--combination_type', type=str, default="VPSDE", help="interpolant type")  
    parser.add_argument('--init_weight', type=float, default=1e-2)  # init
    parser.add_argument('--final_weight', type=float, default=1e-2)  # final
    parser.add_argument('--scale', type=float, default=1., help="scale of loss terms")
    
    # solver parameters
    parser.add_argument('--solver', type=str, default='trapz', choices=SOLVERS)
    parser.add_argument('--quad_step', type=int, default=100)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-5)
    
    # bridge model
    parser.add_argument("--bridge", action='store_true', default=False)
    parser.add_argument("--OT", action='store_true', default=False)
    parser.add_argument('--GRBC_epsilon', type=float, default=1e-5)
    parser.add_argument('--gamma_t', type=float, default=0.1)

    # Gaussian process
    parser.add_argument("--gpp", action='store_true', default=False, help="Gaussian process prior / Gaussian Mixture / Kumaraswamy Mixture")
    parser.add_argument('--grid_size', type=int, default=200)
    parser.add_argument('--K_gmm', type=int, default=5, help="num of components of GMM or KMM")
    parser.add_argument('--constraint_type', type=str, default="spherical", choices=["spherical", "affine"])  
  
    # decomposition
    parser.add_argument("--decomp", action='store_true', default=False)
    parser.add_argument('--K_decomp', type=int, default=400)
    parser.add_argument("--polynomials", type=str, default="gaussianrbf", choices=Pk.POLYNOMIALS)
    parser.add_argument("--noise_type", type=str, default="zero")
    parser.add_argument('--clip_tau', type=float, default=1.)
    parser.add_argument("--auxiliary_loss", action='store_false', default=True)
    parser.add_argument("--bregman_type", type=str, default="logistic")
    parser.add_argument("--shape_type", type=str, default="uniform")
    
    # Engery modeling
    parser.add_argument("--energy", action='store_true', default=False)

    # Secant Alignment Identity
    parser.add_argument("--SAI", action='store_true', default=False)
    parser.add_argument("--t_mode", type=str, default="uniform", choices=["uniform", "IS", "lognorm", "AIS"])
    parser.add_argument("--aligned_ratio", type=float, default=0.5)
    parser.add_argument("--IA", type=eval, default=False, choices=[True, False], help="Interval Annealing")

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
    parser.add_argument('--ood_in_dist', type=str, default='mnist', choices=['mnist', 'fashion_mnist', 'cifar10', 'cifar100'], help='In-distribution dataset for OOD detection')
    parser.add_argument('--ood_base_type', type=str, default='universal', choices=['local', 'general', 'universal'], help='Base distribution type for OOD detection')
    parser.add_argument('--ood_noise_std', type=float, default=0.1, help='Noise standard deviation for OOD detection')
    parser.add_argument('--ood_anneal_alpha_init', type=float, default=0.142857, help='Initial annealing alpha for mixup')
    parser.add_argument('--ood_anneal_gamma', type=float, default=1.0, help='Annealing gamma for alpha')
    parser.add_argument('--ood_crop_size', type=int, default=12, help='Crop size for local OOD detection')
 
    # Image backbone (ResNet or WideResNet for all image-related tasks)
    parser.add_argument('--image_backbone', type=str, default='unet', choices=['unet', 'resnet18', 'resnet34', 'resnet50', 'wideresnet28_10'], help='Backbone network for image-based tasks (OOD detection, real images, etc.)')
    parser.add_argument('--image_dropout', type=float, default=0.3, help='Dropout rate for WideResNet backbone')
    return parser

def set_image_args(args):
    args.GRBC_epsilon = 0.
    args.eval_freq = 10
    args.embed_dim = 128
    args.layer_type = "conv"    # useless now
    args.dims = "128-256-512"  # 128-256-512   or 64-128-256
    args.grad_clip = 10.
    if args.data in ["mnist", "image", "fashion_mnist"]:
        args.batch_size = 1024
        args.test_batch_size = 1024
    else:
        args.batch_size = 512
        args.test_batch_size = 512
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
    args.combination_type = "gpp" if args.gpp else args.combination_type
    args.gpp = True if args.combination_type == "gpp" else args.gpp

    if args.subsub_method == "d3re":
        args.bridge = True
    
    # ISA
    args.IA = True if args.aligned_ratio == 0.15 else args.IA
    args.aligned_ratio = 0.15 if args.IA else args.aligned_ratio
    
    # Energy modeling
    args.joint = False if args.energy else args.joint

    # --condition True --t_mode IS
    if args.task == "likelihood_estimation":
        if args.subtask == "tabular":
            # python main.py likelihood_estimation --subtask tabular --data bsds300 --method dre --decomp --K_decomp 400 --batch_size 20000 --nhidden 5 --hdim_factor 1 --gpu_id 1
            ds_dict = {'swissroll': 2, '8gaussians': 2, 'pinwheel': 2, 'circles': 2, 
               'moons': 2, '2spirals': 2, 'checkerboard': 2, 'rings': 2, 
               'bsds300': 63, 'power': 6, 'gas': 8, 'hepmass': 21, 'miniboone': 43}
            args.d = ds_dict[args.data]
            
            # args.combination_type = "linear"
            args.grad_clip = 5.
            if args.decomp:
                args.hdim_factor = 2
                args.dims = '-'.join([str(64 * args.hdim_factor)] * args.nhidden)
            else:
                args.hdim_factor = 4
                hidden_dim = max(args.hdim_factor * args.d, 128)
                args.dims = '-'.join([str(hidden_dim)] * args.nhidden)
        
        elif args.subtask == "synthetic2d":
            # python main.py likelihood_estimation --subtask synthetic2d --data checkerboard --method dre --decomp --K_decomp 400
            args.specnorm = False
            args.norm = False
            if not args.test_only:
                args.quad_step = 20
        
        elif args.subtask == "real_image":
            # python main.py likelihood_estimation --subtask real_image --data mnist --method dre --decomp --K_decomp 1000 --gpu_id 0
            args = set_image_args(args)
            if isinstance(args.gpu_id, list):
                num_gpus = len(args.gpu_id)
            else:
                num_gpus = 1
            args.early_stopping = 60 * num_gpus
    
    elif args.task == "fdiv_estimation":
        args.eval_freq = 100
        args.num_data_head = 3
        args.num_time_head = 3
        
        ds_dict = {'cpd_text': 20, 'cpd_creditcard': 4}
        args.d = ds_dict.get(args.data, args.d)
        
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
        if args.subtask == "distribution_shift":
            # python main.py fdiv_estimation --method dre --subtask distribution_shift --data gauss_shift --decomp --K_decomp 300 --shift_steps 5 --shift 0.05 --d 80
            args.epochs = 10000
            args.batch_size = 9600
            args.norm = False
            
        elif args.subtask == "change_point_detection":
            # python main.py fdiv_estimation --method dre --subtask change_point_detection --data cpd_text --use_simplex True --cpd_swl 64
            args.dims = "128-128-128"
            args.reload_interval = args.eval_freq * 5
            args.num_data_head = 1
            args.num_time_head = 1
            # args.batch_size = 500
        
        elif args.subtask == "mutual_information":
            # python main.py fdiv_estimation --method dre --subtask mutual_information --data diag_multi_gauss --decomp --K_decomp 1000 --epochs 40000 --d 160 --gpu_id 0
            # python main.py fdiv_estimation --method dre --subtask mutual_information --data half_cube_map --decomp --K_decomp 300 --epochs 10000 --d 2 --rho 0.5 --batch_size 4800 --gpu_id 0
            args.reload_interval = args.eval_freq * 5
        
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
            # args.dims = "128-256-512"  # Same as real_image for CNN
            args.epochs = 500
            # args.batch_size = 512
            args.eval_freq = 10
            args.reload_interval = args.epochs
    
    return args

def get_save_path(args):
    loss_type = "joint" if args.joint else "time"
    key_args = [args.method]
    
    if args.task == "likelihood_estimation":
        if args.sub_method in ["kernel", "neural"]:
            key_args.extend([args.sub_method, args.subsub_method])
        elif args.sub_method == "score":
            key_args.extend([args.combination_type])
        
    elif args.task == "fdiv_estimation":
        if args.sub_method in ["kernel", "neural"]:
            key_args.extend([args.sub_method, args.subsub_method])
            key_args.extend([f"d{args.d}"])
        elif args.sub_method == "score":
            key_args.extend([f"d{args.d}"])
            key_args.extend([args.combination_type])
            
        if args.subtask == "mutual_information":
            key_args.extend(["rho"+str(args.rho)])
        
        if args.subtask == "ood_detection":
            key_args.extend([args.ood_in_dist, args.ood_base_type])
  
    else:
        raise ValueError(f"Task {args.task} is not implemented!")
    
    if args.sub_method == "score":
        if args.bridge:
            key_args.extend(["bridge", str(args.gamma_t)])
    
        if args.decomp:
            auxiliary_loss = "na" if not args.auxiliary_loss else "aux"
            shape_type = None if args.shape_type == "uniform" else args.shape_type
            key_args.extend(["dc", shape_type, args.polynomials, args.noise_type, "K"+str(args.K_decomp), auxiliary_loss])
        elif args.SAI:
            key_args.extend(["sai", args.t_mode, f"ratio{args.aligned_ratio}"])
    
        if args.condition:
            key_args.extend(["cond"])
        if args.OT:
            key_args.extend(["OT"])
        if args.combination_type == "gpp":
            key_args.extend([args.constraint_type, f"K_{args.K_gmm}"])
        key_args.extend([loss_type])
    
    save_path = f'./results/{args.task}/{args.subtask}/' + f'{args.data}/' + '_'.join(filter(None, key_args)) 
    return save_path
