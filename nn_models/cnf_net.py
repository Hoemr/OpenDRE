# -*- coding: utf-8 -*-
"""Continuous normalizing flow network (CNF) using ODE integration."""
import torch
from torchdiffeq import odeint_adjoint as odeint

from dre_utils.utils import _flip
from .tabular_models import JointScoreModel
from nn_models.layers.blocks import ODEfunc


class cnf_net(torch.nn.Module):
    def __init__(self, args, hidden_dims, device='cuda', func=None):
        super(cnf_net, self).__init__()
        self.activation = args.nonlinearity
        self.register_buffer("sqrt_end_time", torch.sqrt(torch.tensor(1.)))
        dim = args.d
        if func is None:
            diffeq = JointScoreModel(
                input_dim=dim,
                hidden_dims=hidden_dims,
                nonlinearity=args.nonlinearity,
                args=args,
            )
            odefunc = ODEfunc(diffeq=diffeq, divergence_fn=args.divergence_fn, residual=args.residual, rademacher=args.rademacher)
            odefunc.before_odeint()
        else:
            odefunc = func
        self.time_deriv_func = odefunc
        self.odefunc = self.time_deriv_func
        self.NFE = 0
        self.solver = args.solver
        self.rtol = args.rtol
        self.atol = args.atol

    def save_state(self, fn='state.tar'):
        torch.save(self.state_dict(), fn)

    def load_state(self, fn='state.tar'):
        self.load_state_dict(torch.load(fn, map_location='cpu'))

    def forward(self, z, delta_logpz=None, integration_times=None, reverse=False):
        device = z.device
        if delta_logpz is None:
            delta_logpz = torch.zeros(z.shape[0], 1).to(device)
        if integration_times is None:
            integration_times = torch.tensor([0.0, 1.0]).to(z)
        if reverse:
            integration_times = _flip(integration_times, 0)
        self.odefunc.before_odeint()
        state = odeint(
            self.time_deriv_func,
            (z, delta_logpz),
            integration_times,
            method=self.solver,
            atol=self.atol,
            rtol=self.rtol,
        )
        self.NFE = self.odefunc._num_evals
        if len(integration_times) == 2:
            state = tuple(s[1] for s in state)
        z, delta_logpz = state
        return z, delta_logpz
