# -*- coding: utf-8 -*-
"""Naming of the per-run result directory: encodes the experiment settings into the path."""


def _add_ood_keys(args, key_args):
    key_args.extend([args.ood_in_dist, args.image_backbone, args.ood_train, f"noise{args.sample_noise_std}", ])
    if args.use_edm_precond:
        key_args.extend(["edm"])
    if getattr(args, "ood_use_imglist", False):
        key_args.append("imglist")
    return key_args


def _add_fdiv_keys(args, key_args):
    key_args.extend([f"d{args.d}"])

    if args.subtask == "mutual_information":
        key_args.extend(["rho" + str(args.rho)])
    elif args.subtask == "ood_detection":
        key_args = _add_ood_keys(args, key_args)
    return key_args


def _add_score_keys(args, key_args, loss_type):
    """Keys contributed by the score-based estimator (--sub_method score)."""
    if args.bridge:
        key_args.extend(["bridge" + str(args.gamma_t)])

    key_args.extend([args.t_mode])

    if args.decomp:
        shape_type = None if args.shape_type == "uniform" else args.shape_type
        key_args.extend([shape_type, args.polynomials, args.noise_type, "K" + str(args.K_decomp)])
    elif args.SAI:
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
    return key_args


def get_save_path(args):
    loss_type = "joint" if args.joint else "time"
    key_args = [args.method, args.sub_method, args.subsub_method]
    if args.sub_method == "score":
        key_args.extend([args.path_type])

    if args.task == "likelihood_estimation":
        pass

    elif args.task == "fdiv_estimation":
        key_args = _add_fdiv_keys(args, key_args)

    else:
        raise ValueError(f"Task {args.task} is not implemented!")

    if args.sub_method == "score":
        key_args = _add_score_keys(args, key_args, loss_type)
    key_args.extend([f"seed{args.seed}"])

    save_path = f'./results/{args.task}/{args.subtask}/' + f'{args.data}/' + '_'.join(filter(None, key_args))
    return save_path
