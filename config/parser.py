# -*- coding: utf-8 -*-
"""CLI entry point: assemble the sub-commands and normalize the parsed namespace."""

import argparse

from .common_args import finalize_common_args
from .tasks.fdiv import finalize_fdiv_args, get_fdiv_args
from .tasks.likelihood import finalize_likelihood_args, get_likelihood_args


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
    args = finalize_common_args(args)

    if args.task == "likelihood_estimation":
        args = finalize_likelihood_args(args)
    elif args.task == "fdiv_estimation":
        args = finalize_fdiv_args(args)

    return args
