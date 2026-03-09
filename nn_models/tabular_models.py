# -*- coding: utf-8 -*-
"""Tabular-data neural network models: joint score and density ratio.

Compared to the previous unified ``UnifiedModel``, these classes only focus on
the architecture and forward logic for tabular tasks. When adding new tabular
tasks in the future, you only need to extend this file instead of touching the
unified image/tabular model.
"""

import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

import layers.layers as Layers
import polynomials.polynomials as Pk
from layers.activations import NONLINEARITIES, INIT_METHODS, ACTIVATION_TYPE_MAP


class _BaseTabularModel(nn.Module):
    """Shared utilities for tabular models (EDM preconditioning, init, etc.)."""

    def __init__(self, args=None, path=None):
        super().__init__()
        self.args = args
        self.path = path
        self.joint = args.joint if args is not None else False

        # EDM 相关配置（与 UnifiedModel 保持一致，便于行为对齐）
        self.use_edm_precond = getattr(args, "use_edm_precond", False)
        self.edm_sigma_data = getattr(args, "edm_sigma_data", 1.0)
        self.edm_sigma_min = getattr(args, "edm_sigma_min", 1e-3)
        self.edm_use_noise_embed = getattr(args, "edm_use_noise_embed", False)
        self.edm_debug = getattr(args, "edm_debug", False)
        self.register_buffer("edm_v_src", None)
        self.register_buffer("edm_v_dst", None)

        self.noise_embed = None
        self.nfe = 0  # number of function evaluations

    # --------------------
    # Initialization and activations
    # --------------------
    def _get_activation(self, nonlinearity):
        """Return activation function by name (mirrors UnifiedModel._get_activation)."""
        if nonlinearity in NONLINEARITIES.keys():
            if nonlinearity == "dyt":
                # Keep the same logic as in UnifiedModel
                if getattr(self, "model_type", None) == "decomp_joint_score":
                    input_dim = 4 if getattr(self, "data_type", None) == "image" else 2
                else:
                    input_dim = 2
                K = getattr(self, "K", 1)
                return NONLINEARITIES[nonlinearity](num_features=K, input_dim=input_dim)
            return NONLINEARITIES[nonlinearity]
        else:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")

    def _detect_activation(self, layer):
        """Detect activation type from submodules (aligned with UnifiedModel)."""
        for _, child in itertools.islice(layer.named_children(), 3):
            if type(child) in ACTIVATION_TYPE_MAP:
                return ACTIVATION_TYPE_MAP[type(child)]
        return "identity"

    def _initialize_weights(self):
        """Initialize network weights (copied from UnifiedModel for consistency)."""
        for _, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                act_name = self._detect_activation(module)
                INIT_METHODS.get(act_name, INIT_METHODS["identity"])(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.GroupNorm):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)

    # --------------------
    # EDM preconditioning
    # --------------------
    def set_edm_endpoint_stats(self, v_src, v_dst):
        """Set EDM endpoint statistics (keep the same API as UnifiedModel)."""
        dev = next(self.parameters()).device
        self.register_buffer("edm_v_src", v_src.to(dev))
        self.register_buffer("edm_v_dst", v_dst.to(dev))

    def _edm_precond_input(self, t, x):
        """EDM input scaling (copied from UnifiedModel._edm_precond_input)."""
        if not self.use_edm_precond or self.path is None:
            return x, None

        s_t2 = self.path.total_scale_squared_t_only(t)
        s_t = torch.sqrt(s_t2.clamp(min=1e-10)).clamp(min=self.edm_sigma_min)
        c_in = 1.0 / torch.sqrt(self.edm_sigma_data ** 2 + s_t ** 2)
        if x.dim() == 4:
            c_in = c_in.view(-1, 1, 1, 1)
        elif x.dim() == 2:
            c_in = c_in.view(-1, 1)
        x_scaled = c_in * x
        c_noise = None
        if self.edm_use_noise_embed and self.noise_embed is not None:
            s_t_scalar = s_t
            if s_t.dim() == 2 and s_t.shape[1] > 1:
                s_t_scalar = torch.sqrt((s_t ** 2).mean(dim=1, keepdim=True))
            c_noise = 0.25 * torch.log(s_t_scalar.clamp(min=1e-10))
        if self.edm_debug:
            with torch.no_grad():
                path_type = "bridge" if getattr(self.path, "bridge", False) else "default"
                noise_embed_on = self.edm_use_noise_embed and self.noise_embed is not None
                st_mean, st_std = s_t.float().mean().item(), s_t.float().std().item()
                st_min, st_max = s_t.float().min().item(), s_t.float().max().item()
                x_mean, x_std = x.float().mean().item(), x.float().std().item()
                xs_mean, xs_std = x_scaled.float().mean().item(), x_scaled.float().std().item()
                import logging

                logging.getLogger(__name__).info(
                    f"EDM_precond: path={path_type} noise_embed={noise_embed_on} | "
                    f"s_t mean={st_mean:.4f} std={st_std:.4f} min={st_min:.4f} max={st_max:.4f} | "
                    f"x_t raw mean={x_mean:.4f} std={x_std:.4f} | x_t^in mean={xs_mean:.4f} std={xs_std:.4f}"
                )
        return x_scaled, c_noise

    # --------------------
    # Decomposition-related helpers (used by decomp_joint_score models)
    # --------------------
    def set_fk(self, f_k=None):
        self.fk = f_k

    def get_fk_reg(self, x=None):
        if not hasattr(self, "fk"):
            return None
        if x is not None or self.fk is None:
            return self.compute_fk(x)
        else:
            return self.fk

    def compute_fk(self, x):
        return self.compute_fk_impl(x)

    def compute_weighted_sum(self, x, f_k=None):
        if f_k is None:
            f_k = self.compute_fk(x)
        weighted_sum = torch.sum(f_k * self.Pk.integrals, dim=-1, keepdim=True)
        return weighted_sum.squeeze()

    def time_gradient(self, t, x, f_k=None):
        if f_k is None:
            f_k = self.compute_fk(x)
        g_k = self.Pk.grad_function(t)
        return torch.sum(f_k * g_k, dim=-1, keepdim=True)

    # --------------------
    # Misc
    # --------------------
    def reset_states(self):
        self.nfe = 0


class _BaseTabularTimeDependentModel(_BaseTabularModel):
    """Base class for time-dependent tabular models (joint_score / decomp_joint_score)."""

    def _build_tabular_architecture(
        self,
        input_dim,
        hidden_dims,
        embed_dim,
        nonlinearity,
        args,
        norm,
        use_spectral_norm,
        dynamic,
    ):
        """Tabular architecture builder extracted from UnifiedModel._build_tabular_architecture."""
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        dropout = getattr(args, "dropout", 0.0) if args else 0.0
        pre_norm = getattr(args, "pre_norm", False) if args else False
        timeemb_outdim = input_dim

        def build_dynamic_layers(
            input_dim=None,
            output_dim=None,
            num_layers=None,
            dim_sequence=None,
            activation="leakyrelu",
            max_expansion=4.0,
            min_layers=1,
        ):
            if dim_sequence is None:
                if dynamic:
                    if num_layers is None:
                        if input_dim == output_dim:
                            num_layers = min_layers
                        else:
                            ratio = output_dim / input_dim
                            num_layers = max(
                                min_layers,
                                int(np.ceil(np.log(ratio) / np.log(max_expansion))),
                            )

                    dim_sequence = list(
                        np.geomspace(
                            input_dim,
                            output_dim,
                            num=num_layers + 1,
                            dtype=int,
                        )
                    )
                else:
                    dim_sequence = [input_dim] + [input_dim] * num_layers + [output_dim]

            layers = []
            for i in range(len(dim_sequence) - 1):
                in_dim, out_dim = dim_sequence[i], dim_sequence[i + 1]
                if getattr(self, "model_type", None) == "joint_score":
                    # For joint_score models, use time-aware residual blocks
                    layers.append(
                        Layers.ResidualBlockWithTime(
                            in_dim,
                            out_dim,
                            embed_dim=timeemb_outdim,
                            activation=self._get_activation(activation),
                            dropout=dropout,
                            norm=norm,
                            use_spectral_norm=use_spectral_norm,
                            pre_norm=pre_norm,
                        )
                    )
                else:
                    # For other models, use standard residual blocks
                    layers.append(
                        Layers.ResidualBlock(
                            in_dim,
                            out_dim,
                            activation=self._get_activation(activation),
                            dropout=dropout,
                            norm=norm,
                            use_spectral_norm=use_spectral_norm,
                            pre_norm=pre_norm,
                        )
                    )
            return layers

        # Shared network
        dim_sequence = [input_dim] + hidden_dims
        self.shared_layers = nn.Sequential(
            *build_dynamic_layers(dim_sequence=dim_sequence, activation=nonlinearity)
        )

        # Time head
        if getattr(self, "model_type", None) == "decomp_joint_score":
            # Decomposition models output K components
            self.last_layer_time = Layers.LinearLastLayer(hidden_dims[-1], self.K)
            self.time_head = nn.Sequential(
                *build_dynamic_layers(
                    input_dim=hidden_dims[-1],
                    output_dim=hidden_dims[-1],
                    num_layers=getattr(args, "num_time_head", 1) - 1,
                    activation=nonlinearity,
                ),
                self.last_layer_time,
            )
        else:
            # Other models output a scalar
            output_dim = 1
            self.last_layer_time = Layers.LinearLastLayer(hidden_dims[-1], output_dim)
            self.time_head = nn.Sequential(
                *build_dynamic_layers(
                    input_dim=hidden_dims[-1],
                    output_dim=hidden_dims[-1],
                    num_layers=getattr(args, "num_time_head", 1) - 1,
                    activation=nonlinearity,
                ),
                self.last_layer_time,
            )

        # Data head: only used when self.joint is True
        if self.joint and getattr(self, "model_type", None) in [
            "joint_score",
            "decomp_joint_score",
        ]:
            if getattr(self, "model_type", None) == "joint_score":
                last_layer_data = Layers.LinearLastLayer(hidden_dims[-1], input_dim)
                self.data_head = nn.Sequential(
                    *build_dynamic_layers(
                        input_dim=hidden_dims[-1],
                        output_dim=hidden_dims[-1],
                        num_layers=getattr(args, "num_time_head", 1) - 1,
                        activation=nonlinearity,
                    ),
                    last_layer_data,
                )
            else:  # decomp_joint_score
                last_layer_data = Layers.LinearLastLayer(hidden_dims[-1], input_dim)
                self.data_head = nn.Sequential(
                    *[
                        Layers.ResidualBlockWithTime(
                            in_dim=hidden_dims[-1],
                            out_dim=hidden_dims[-1],
                            embed_dim=timeemb_outdim,
                            activation=self._get_activation(nonlinearity),
                            dropout=dropout,
                            norm=norm,
                            use_spectral_norm=use_spectral_norm,
                            pre_norm=pre_norm,
                        )
                        for _ in range(getattr(args, "num_data_head", 1) - 1)
                    ],
                    last_layer_data,
                )

        # Time embedding and SAI fusion
        self.time_emb = Layers.FourierSinusoidalTimeEncoder(
            output_dim=timeemb_outdim, nfreq=embed_dim
        )
        if getattr(args, "SAI", False) if args else False:
            self.time_fusion = nn.Sequential(
                nn.Linear(timeemb_outdim * 2, timeemb_outdim),
                nn.SiLU(),
                nn.Linear(timeemb_outdim, timeemb_outdim),
            )

    # ---- s_t computation: implemented by subclasses via _compute_s_t ----
    def _compute_s_t(self, t, x, time_emb):
        raise NotImplementedError

    def _forward_time_dependent(self, t, x, cond=None, joint=True):
        """Forward pass for time-dependent tabular models

        This is essentially UnifiedModel._forward_tabular_time_dependent,
        specialized to tabular-only models.
        """
        self.nfe += 1
        if self.use_edm_precond:
            x, c_noise = self._edm_precond_input(t, x)
        else:
            c_noise = None

        if self.time_emb is not None:
            time_emb = self.time_emb(t)  # [B, timeemb_outdim]
            if c_noise is not None and self.noise_embed is not None:
                time_emb = time_emb + self.noise_embed(c_noise)
            if cond is not None:
                rel_time_emb = self.time_emb(t - cond)
                concat_emb = torch.cat([time_emb, rel_time_emb], dim=-1)
                time_emb = (
                    self.time_fusion(concat_emb)
                    if hasattr(self, "time_fusion")
                    else time_emb
                )
        else:
            time_emb = None

        # shared layers
        for layer in self.shared_layers:
            if hasattr(layer, "forward"):
                if hasattr(layer, "embed_dim"):
                    x = layer(x, time_emb=time_emb)
                else:
                    x = layer(x)
            else:
                x = layer(x)

        s_t = self._compute_s_t(t, x, time_emb)

        if self.joint and joint and hasattr(self, "data_head"):
            s_x = x
            for layer in self.data_head:
                if hasattr(layer, "embed_dim"):
                    s_x = layer(s_x, time_emb=time_emb)
                else:
                    s_x = layer(s_x)
            return s_x, s_t
        return s_t

    def forward(self, t, x, cond=None, joint=True):
        return self._forward_time_dependent(t, x, cond=cond, joint=joint)


class JointScoreModel(_BaseTabularTimeDependentModel):
    """Tabular joint score model."""

    def __init__(
        self,
        input_dim,
        hidden_dims,
        embed_dim=256,
        nonlinearity="leakyrelu",
        args=None,
        norm=False,
        use_spectral_norm=False,
        dynamic=False,
        path=None,
    ):
        super().__init__(args=args, path=path)
        self.model_type = "joint_score"
        self.data_type = "tabular"
        # K is not used for joint_score, but keeps the API consistent for dyt activation
        self.K = 1

        self._build_tabular_architecture(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            embed_dim=embed_dim,
            nonlinearity=nonlinearity,
            args=args,
            norm=norm,
            use_spectral_norm=use_spectral_norm,
            dynamic=dynamic,
        )

        # Optional EDM noise embedding
        if self.use_edm_precond and self.edm_use_noise_embed:
            time_emb_dim = input_dim
            self.noise_embed = nn.Sequential(
                nn.Linear(1, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )

        self._initialize_weights()

    def _compute_s_t(self, t, x, time_emb):
        """Same as UnifiedModel._compute_s_t_tabular_joint."""
        s_t = x
        for layer in self.time_head:
            if hasattr(layer, "forward"):
                if hasattr(layer, "embed_dim"):
                    s_t = layer(s_t, time_emb=time_emb)
                else:
                    s_t = layer(s_t)
            else:
                s_t = layer(s_t)
        return s_t


class DecompJointScoreModel(_BaseTabularTimeDependentModel):
    """Tabular decomp joint score model with polynomial decomposition."""

    def __init__(
        self,
        input_dim,
        hidden_dims,
        embed_dim=256,
        K=300,
        polynomials="gaussrbf",
        noise_type="zero",
        nonlinearity="leakyrelu",
        args=None,
        norm=False,
        use_spectral_norm=False,
        dynamic=False,
        path=None,
    ):
        super().__init__(args=args, path=path)
        self.model_type = "decomp_joint_score"
        self.data_type = "tabular"
        self.K = K

        # Polynomial decomposition basis, same as in UnifiedModel
        self.Pk = Pk.PerturbedFrame(
            K,
            noise_type=noise_type,
            polynomials=polynomials,
            shape_type=getattr(args, "shape_type", "uniform"),
        )

        self._build_tabular_architecture(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            embed_dim=embed_dim,
            nonlinearity=nonlinearity,
            args=args,
            norm=norm,
            use_spectral_norm=use_spectral_norm,
            dynamic=dynamic,
        )

        # f_k computation function (copied from UnifiedModel._compute_fk_tabular)
        self.compute_fk_impl = self._compute_fk_tabular

        if self.use_edm_precond and self.edm_use_noise_embed:
            time_emb_dim = input_dim
            self.noise_embed = nn.Sequential(
                nn.Linear(1, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )

        self._initialize_weights()

    def _compute_fk_tabular(self, x):
        """Compute f_k features for tabular data."""
        x = self.shared_layers(x)
        return self.time_head(x)  # (..., K)

    def _compute_s_t(self, t, x, time_emb):
        """Same as UnifiedModel._compute_s_t_tabular_decomp."""
        f_k = self.time_head(x)
        g_k = self.Pk(t)
        s_t = torch.sum(f_k * g_k, dim=-1, keepdim=True)

        # Numerical stability: avoid NaNs in gradients
        f_k = torch.clamp(f_k, min=-1e6, max=1e6)
        g_k = torch.sign(g_k) * torch.clamp(torch.abs(g_k), min=1e-8)

        self.set_fk(f_k)

        if torch.isnan(s_t).any().item():
            print(
                "f_k:",
                f_k.mean(dim=0),
                "\n g_k:",
                g_k.mean(dim=0),
                "\n s_t: ",
                s_t.mean(dim=0),
            )
            raise ValueError("NaN detected in time score computation")
        return s_t


class NeuralDensityRatioModel(_BaseTabularModel):
    """Tabular neural density-ratio model."""

    def __init__(
        self,
        input_dim,
        hidden_dims,
        nonlinearity="leakyrelu",
        args=None,
        norm=False,
        use_spectral_norm=False,
        dynamic=False,
    ):
        super().__init__(args=args, path=None)
        self.model_type = "density_ratio"
        self.data_type = "tabular"

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        dropout = getattr(args, "dropout", 0.0) if args else 0.0
        pre_norm = getattr(args, "pre_norm", False) if args else False

        def build_dynamic_layers(
            input_dim=None,
            output_dim=None,
            num_layers=None,
            dim_sequence=None,
            activation="leakyrelu",
            max_expansion=4.0,
            min_layers=1,
        ):
            if dim_sequence is None:
                if dynamic:
                    if num_layers is None:
                        if input_dim == output_dim:
                            num_layers = min_layers
                        else:
                            ratio = output_dim / input_dim
                            num_layers = max(
                                min_layers,
                                int(np.ceil(np.log(ratio) / np.log(max_expansion))),
                            )
                    dim_sequence = list(
                        np.geomspace(
                            input_dim,
                            output_dim,
                            num=num_layers + 1,
                            dtype=int,
                        )
                    )
                else:
                    dim_sequence = [input_dim] + [input_dim] * num_layers + [output_dim]

            layers = []
            for i in range(len(dim_sequence) - 1):
                in_dim, out_dim = dim_sequence[i], dim_sequence[i + 1]
                layers.append(
                    Layers.ResidualBlock(
                        in_dim,
                        out_dim,
                        activation=self._get_activation(activation),
                        dropout=dropout,
                        norm=norm,
                        use_spectral_norm=use_spectral_norm,
                        pre_norm=pre_norm,
                    )
                )
            return layers

        # Shared network
        dim_sequence = [input_dim] + hidden_dims
        self.shared_layers = nn.Sequential(
            *build_dynamic_layers(dim_sequence=dim_sequence, activation=nonlinearity)
        )

        # Final scalar output layer
        self.last_layer_time = Layers.LinearLastLayer(hidden_dims[-1], 1)
        self.time_head = nn.Sequential(
            *build_dynamic_layers(
                input_dim=hidden_dims[-1],
                output_dim=hidden_dims[-1],
                num_layers=getattr(args, "num_time_head", 1) - 1,
                activation=nonlinearity,
            ),
            self.last_layer_time,
        )

        self._initialize_weights()

    def _forward_density_ratio(self, x):
        for layer in self.shared_layers:
            x = layer(x)

        for layer in self.time_head:
            x = layer(x)

        return x

    def forward(self, x, cond=None, joint=True):
        return self._forward_density_ratio(x)

