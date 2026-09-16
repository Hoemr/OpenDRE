# -*- coding: utf-8 -*-
"""Default overrides shared by the image-based tasks (real images, MI on real data, OOD detection)."""


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
