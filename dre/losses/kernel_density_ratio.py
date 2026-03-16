# -*- coding: utf-8 -*-
"""Kernel-based density ratio estimation training step."""
import torch
import torch.nn.functional as F


class KernelDensityRatioTrainStepFn:
    """
    Kernel-based density ratio estimation training step function.
    """
    def __init__(self, args, path_model=None, method='kulsif'):
        self.args = args
        self.method = method
        self.reg_param = getattr(args, 'reg_param', 0.1)
        self.path = path_model

    def __call__(self, model, batch, step=None):
        qx, px = batch
        model.collect_data(qx, px)
        if model.is_trained:
            loss, nll = self._compute_loss(model, qx, px)
        else:
            loss = torch.tensor(0.0, requires_grad=False)
            nll = torch.tensor(0.0, requires_grad=False)
        return {'loss': loss, 'nll': nll}

    def _compute_loss(self, model, qx, px):
        log_r_qx = torch.log(model(qx) + 1e-8)
        log_r_px = torch.log(model(px) + 1e-8)
        if self.method == 'kulsif':
            r_qx = torch.exp(log_r_qx)
            r_px = torch.exp(log_r_px)
            loss = 0.5 * torch.mean(r_qx**2) - torch.mean(r_px)
        elif self.method == 'lr':
            loss = (torch.mean(F.softplus(log_r_qx)) + torch.mean(F.softplus(-log_r_px)))
        elif self.method == 'kliep':
            r_qx = torch.exp(log_r_qx)
            loss = -torch.mean(log_r_px) + torch.mean(r_qx)
        elif self.method == 'exp':
            loss = (torch.mean(torch.exp(torch.clamp(log_r_qx, max=10))) - torch.mean(log_r_px))
        elif self.method == 'sq':
            r_qx = torch.exp(torch.clamp(log_r_qx, min=-5., max=5.))
            r_px = torch.exp(torch.clamp(log_r_px, min=-5., max=5.))
            loss = torch.mean(r_qx**2) - 2 * torch.mean(r_px)
        nll = -log_r_px.mean()
        if hasattr(self.path, 'prior_logp'):
            prior_terms = self.path.prior_logp(px)
            nll = -(log_r_px + prior_terms).mean()
        return loss, nll
