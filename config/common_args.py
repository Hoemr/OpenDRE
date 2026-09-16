# -*- coding: utf-8 -*-
"""Arguments shared by every task, plus the task-independent post-parse normalization.

The `_add_*_args` helpers only declare arguments; they are grouped by what the
arguments control (network, training, method-specific features, ...) so that a
feature can be found without scanning one long list.
"""

import torch

from nn_models.layers.activations import NONLINEARITIES
import dre.polynomials as Pk

from .constants import SOLVERS


def _add_network_args(parser):
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


def _add_training_args(parser):
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


def _add_model_args(parser):
    parser.add_argument("--eps", type=float, default=1e-5, help="t \in [eps, 1-eps]")
    parser.add_argument("--joint", type=eval, default=True, choices=[True, False])
    parser.add_argument("--condition", type=eval, default=False, choices=[True, False])
    parser.add_argument('--path_type', type=str, default="VP", help="interpolant type")
    parser.add_argument('--relative_ratio', type=float, default=0.2, help="relative ratio for relative ratio sampling")
    parser.add_argument('--scale', type=float, default=1., help="scale of loss terms")


def _add_solver_args(parser):
    parser.add_argument('--solver', type=str, default='trapz', choices=SOLVERS)
    parser.add_argument('--quad_step', type=int, default=100)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-5)


def _add_bridge_args(parser):
    parser.add_argument("--bridge", action='store_true', default=False)
    parser.add_argument("--OT", action='store_true', default=False)
    parser.add_argument('--sample_noise_std', type=float, default=0.005, help='Std of additive Gaussian noise for sample perturbation')
    parser.add_argument('--gamma_t', type=float, default=0.01)


def _add_edm_precond_args(parser):
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


def _add_mvp_args(parser):
    # Minimum Variance Path
    parser.add_argument("--mvp", action='store_true', default=False, help="Minimum Variance Path / Gaussian Mixture / Kumaraswamy Mixture")
    parser.add_argument('--grid_size', type=int, default=200)
    parser.add_argument('--K_gmm', type=int, default=5, help="num of components of GMM or KMM")
    parser.add_argument('--constraint_type', type=str, default="spherical", choices=["spherical", "affine"])


def _add_decomp_args(parser):
    # decomposition
    parser.add_argument("--decomp", action='store_true', default=False)
    parser.add_argument('--K_decomp', type=int, default=400)
    parser.add_argument("--polynomials", type=str, default="imqrbf", choices=Pk.POLYNOMIALS)
    parser.add_argument("--noise_type", type=str, default="zero")
    parser.add_argument('--clip_tau', type=float, default=1.)
    parser.add_argument("--auxiliary_loss", action='store_false', default=True)
    parser.add_argument("--auxiliary_quad_steps", type=int, default=5,
                        help="Quadrature steps for log r(x) in auxiliary Bregman loss (non-decomp); kept small for training.")
    parser.add_argument("--bregman_weight", type=float, default=1.0,
                        help="Weight (coefficient) in front of the auxiliary Bregman loss; use for ablation (default 1.0).")
    parser.add_argument("--bregman_type", type=str, default="logistic")
    parser.add_argument("--shape_type", type=str, default="uniform")


def _add_energy_args(parser):
    # Engery modeling
    parser.add_argument("--energy", action='store_true', default=False)


def _add_sai_args(parser):
    # Secant Alignment Identity
    parser.add_argument("--SAI", action='store_true', default=False)
    parser.add_argument("--t_mode", type=str, default="uniform", choices=["uniform", "IS", "lognorm", "AIS"])
    parser.add_argument("--aligned_ratio", type=float, default=0.5)
    parser.add_argument("--IA", type=eval, default=False, choices=[True, False], help="Interval Annealing")


def _add_sampling_args(parser):
    # sampling
    parser.add_argument("--sampling_only", action='store_true', default=False)
    parser.add_argument("--test_only", action='store_true', default=False)


def get_common_args(parser):
    _add_network_args(parser)
    _add_training_args(parser)
    _add_model_args(parser)
    _add_solver_args(parser)
    _add_bridge_args(parser)
    _add_edm_precond_args(parser)
    _add_mvp_args(parser)
    _add_decomp_args(parser)
    _add_energy_args(parser)
    _add_sai_args(parser)
    _add_sampling_args(parser)
    return parser


def _resolve_path_type(args):
    # MinPV
    args.path_type = "mvp" if args.mvp else args.path_type
    args.mvp = True if args.path_type == "mvp" else args.mvp
    return args


def _resolve_method_flags(args):
    """Keep `--decomp`/`--bridge`/`--IA`/`--energy` consistent with the chosen variant."""
    if args.decomp:
        args.sub_method = "score"
        args.subsub_method = "osdre"

    if args.subsub_method == "dre_infty":
        args.sub_method = "score"
        args.bridge = False
        args.decomp = False
    elif args.subsub_method == "d3re":
        args.sub_method = "score"
        args.bridge = True
    elif args.subsub_method == "osdre":
        args.sub_method = "score"
        args.decomp = True

    # ISA
    if args.subsub_method == "isadre":
        args.sub_method = "score"
        args.IA = True
        args.aligned_ratio = 0.15
    args.IA = True if args.aligned_ratio == 0.15 else args.IA
    args.aligned_ratio = 0.15 if args.IA else args.aligned_ratio
    if args.IA or args.aligned_ratio == 0.15:
        args.sub_method = "score"
        args.subsub_method = "isadre"

    # Energy modeling
    args.joint = False if args.energy else args.joint
    return args


def finalize_common_args(args):
    """Task-independent post-processing of the parsed namespace."""
    args.device = torch.device(f"cuda:{args.gpu_id[0]}" if torch.cuda.is_available() else "cpu")
    args.test_only = True if args.sampling_only else args.test_only
    args.dropout = 0 if args.condition else args.dropout

    args = _resolve_path_type(args)
    args = _resolve_method_flags(args)
    return args
