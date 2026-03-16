# -*- coding: utf-8 -*-
import torch
from scipy import integrate
from torchdiffeq import odeint
import torchquad
from functools import partial
import torch.nn.functional as F
from torch.autograd import grad
from torch.autograd.functional import jvp

SOLVER_DICT_SCIPY = {
    'dopri5_scipy': 'DOP853', 
    "bdf_scipy": "BDF",
    "rk4_scipy": 'RK45',
}   # used for scipy.integrate
SOLVER_DICT = {
    'dopri5': 'dopri5',
    'rk4': 'rk4',
    'euler': 'euler',
    'midpoint': 'midpoint',
}
QUAD_DICT = {
    'quad': 'quad',   # used for scipy.integrate
    'mc': 'mc',
    'trapz': 'trapz',
    'simpson': 'simpson',
}

def stopgrad(x):
    return x.detach()

def secant_to_tangent(u_theta, xt, dx_dt, l, t, joint=False):
    """
    Compute tangent function s_t(xt,t) from secant model u_theta using Secant Alignment Identity (SAI)
    
    Args:
        u_theta: Secant model (input: xt, s, t → output: u(xt,s,t))
        xt: Interpolated points [batch, dim]
        dx_dt: Derivative of xt w.r.t. t [batch, dim]
        l: Start times [batch, 1] 
        t: End times [batch, 1]
    
    Returns:
        s_t: Tangent function values [batch, 1]
    
    Math:
        s_t(xt,t) = u(xt,l,t) + (t-l) * [ (dx_t/dt)·∂u/∂x + (dt/dt)·∂u/∂t + (dl/dt)·∂u/∂l ]
    """
    # Forward pass through secant model
    if joint:
        s_x, u = u_theta(t, xt, l) # [batch, 1]
    else:
        u = u_theta(t, xt, l)  # [batch, 1]
    
    # Compute Jacobian components, [batch, dim], [batch, 1], [batch, 1]
    _, dudt = jvp(lambda xt, l, t: u, 
                  inputs=(xt, l, t),
                  v=(dx_dt, torch.zeros_like(l), torch.ones_like(t)),
                  create_graph=True)
    
    # Apply Secant Alignment Identity
    s_t = u + stopgrad(torch.clamp(t - l, min=0.0, max=1.0) * dudt)

    if joint:
        return s_x, s_t
    return s_t

def get_likelihood_fn(path_model, args, quad_step=100):
    density_ratio_fn = get_density_ratio_fn(path_model, args, quad_step)

    def likelihood_fn(score_model, x, joint=True, steps=quad_step):
        log_p1_p0, nfe = density_ratio_fn(score_model, x, joint, steps=steps)
        log_p0 = path_model.prior_logp(x)
        assert log_p1_p0.shape == log_p0.shape
        log_p1 = log_p1_p0 + log_p0
        return log_p1, nfe 

    def dc_likelihood_fn(score_model, x, joint=True, steps=quad_step):
        log_p1_p0 = score_model.compute_weighted_sum(x)
        log_p0 = path_model.prior_logp(x)
        assert log_p1_p0.shape == log_p0.shape
        log_p1 = log_p1_p0 + log_p0
        nfe = 1
        return log_p1, nfe 

    if args.decomp:
        return dc_likelihood_fn
    else:
        return likelihood_fn

def get_density_ratio_fn(path_model, args, quad_step=100):
    rtol=args.rtol
    atol=args.atol
    solver=args.solver
    eps=args.eps
    sub_method = args.sub_method
    decomp=args.decomp
    secant=args.SAI
    
    # Quadrature via torchquad
    integrator = None
    if solver == 'mc':
        integrator = torchquad.MonteCarlo()
    elif solver == 'trapz':
        integrator = torchquad.Trapezoid()
    elif solver == 'simpson':
        integrator = torchquad.Simpson()
        
    def integrand(t, y, x, score_model):
            t_tensor = torch.full((x.size(0), 1), t, device=x.device)
            return score_model(t_tensor, x, joint=False)  # shape: [batch_size, 1] or [batch_size]
    
    if solver in SOLVER_DICT:  # ODE solver via torchdiffeq
        def ode_func(t, y, x, score_model):
            return integrand(t, y, x, score_model).view(-1)
        def density_ratio_fn(score_model, x, joint=False, steps=quad_step):
            batch_size = x.size(0)
            p_get_rx = partial(ode_func, x=x, score_model=score_model)
            y0 = torch.zeros(batch_size, device=x.device)
            ts = torch.tensor([0.0, 1.0 - eps], device=x.device)
            score_model.reset_states()
            solution = odeint(p_get_rx, y0, ts, method=SOLVER_DICT[solver], rtol=rtol, atol=atol)
            density_ratio = solution[-1]
            nfe = score_model.nfe  
            return density_ratio, nfe 

    elif solver in QUAD_DICT:
        def batch_integrand(t, x, score_model):
            # t shape: [num_points, dim], flatten to [num_points]
            t = t.squeeze(-1)  # [num_points]
            results = []
            for t_scalar in t:
                val = integrand(t_scalar, 0, x, score_model)
                results.append(val.view(-1))  # [batch_size]
            return torch.stack(results, dim=0)  # [num_points, batch_size]
        
        def density_ratio_fn(score_model, x, joint=False, steps=quad_step):
            limits = [[0.0, 1.0 - eps]]  # 1D integral
            p_get_rx = partial(batch_integrand, x=x, score_model=score_model)
            density_ratio = integrator.integrate(
                p_get_rx,
                dim=1,
                N=steps,
                integration_domain=limits
            )  # shape: [batch_size]
            nfe = steps
            return density_ratio, nfe 
    else:
        raise ValueError(f"Solver {solver} not recognized.")
  
    def dc_density_ratio_fn(score_model, x, joint=False, steps=quad_step):
        log_p1_p0 = score_model.compute_weighted_sum(x)
        nfe = 1
        return log_p1_p0, nfe  

    def secant_density_ratio_fn(score_model, x, joint=False, steps=quad_step):
        batch_size = x.shape[0]
        device = x.device
        
        assert steps >= 1
        
        if joint:
            score_model_t = lambda t, x, l: score_model(t, x, l)[1]
        else:
            score_model_t = lambda t, x, l: score_model(t, x, l)
            
        log_p1_p0 = torch.zeros(batch_size, device=device)
        nfe = 0
        
        if steps == 1:
            # One-step estimation: t=1, l=0
            t = torch.ones(batch_size, 1, device=device)
            l = torch.zeros(batch_size, 1, device=device)
            delta_u = score_model_t(t, x, l)
            log_p1_p0 += delta_u.squeeze() * (t - l).squeeze()  # (t - l) = 1
            nfe = 1
        else:
            # Multi-step case: use transformed time grid
            ts = torch.linspace(0, 1, steps + 1, device=device) 
            logit_t = torch.logit(ts.clamp(eps, 1 - eps))  # transform to logit space
            ts = torch.sigmoid(logit_t * 1.5 + 0.2)  # prefer boundary area
        
            for i in range(steps):
                l = ts[i].expand(batch_size, 1)
                t = ts[i + 1].expand(batch_size, 1)
                
                delta_u = score_model_t(t, x, l)
                delta_log_ratio = delta_u * (t - l)
                
                log_p1_p0 += delta_log_ratio.squeeze() 
                nfe += 1
        
        # # boundary refinement
        # if steps == 1:
        #     t_mid = (ts[0] + ts[1]) / 2
        #     u_mid = score_model(t_mid.expand(batch_size, 1), x, ts[0].expand(batch_size, 1))
        #     correction = 0.5 * (u_mid.squeeze() - delta_u.squeeze()) * (t - l).squeeze()
        #     log_p1_p0 += correction
        #     nfe += 1
        
        return log_p1_p0.squeeze(), nfe
    
    def kernel_density_ratio_fn(model, x, joint=False, steps=quad_step):
        density_ratios = model(x)
        log_density_ratios = torch.log(density_ratios + 1e-8)
        return log_density_ratios, 0  
    
    def neural_density_ratio_fn(model, x, joint=False, steps=quad_step):
        log_p1_p0 = model(x)
        nfe = 1
        return log_p1_p0.squeeze(), nfe  
    
    def energy_density_ratio_fn(model, x, joint=False, steps=quad_step):
        batch_size = x.size(0)
        t0 = torch.tensor(0.0, device=x.device).expand(batch_size, 1)
        t1 = torch.tensor(1.0-eps, device=x.device).expand(batch_size, 1)
        log_p1_p0 = -model(t1, x) + model(t0, x)   # log r(x) = -U(x,1) + U(x,0)
        nfe = 2
        return log_p1_p0.squeeze(), nfe
    
    if sub_method == "kernel":
        return kernel_density_ratio_fn
    elif sub_method == "neural":
        if args.subsub_method == "ew":
            def ew_density_ratio_fn(model, x, joint=False, steps=quad_step):
                raw_f_px = model(x)  
                f_px = F.softplus(raw_f_px) + 1e-8
                log_p1_p0 = 0.5 * torch.log(2 * f_px)
                nfe = 1
                return log_p1_p0.squeeze(), nfe
            return ew_density_ratio_fn
        elif args.subsub_method == "pw":
            def pw_density_ratio_fn(model, x, joint=False, steps=quad_step, k=1.0):
                f_px = F.softplus(model(x)) + 1e-8
                estimated_ratio = ((1 + k) * f_px) ** (1 / (1 + k))
                log_p1_p0 = torch.log(estimated_ratio + 1e-8)
                nfe = 1
                return log_p1_p0.squeeze(), nfe
            return pw_density_ratio_fn

        return neural_density_ratio_fn
    else:   # score-based density ratio
        if decomp:
            return dc_density_ratio_fn
        elif secant:
            return secant_density_ratio_fn
        elif args.energy:
            return energy_density_ratio_fn  
        else:
            return density_ratio_fn

class MetricEvaluator:
    def __init__(self, eps=1e-8):
        self.eps = eps  # avoid divide zero

    def relative_error(self, y_true, y_pred):
        """|pred - true| / |true|"""
        return torch.mean(torch.abs(y_pred - y_true) / (torch.abs(y_true) + self.eps))

    def mae(self, y_true, y_pred):
        """|pred - true|"""
        return torch.mean(torch.abs(y_pred - y_true))

    def mse(self, y_true, y_pred):
        """(pred - true)^2"""
        return torch.mean((y_pred - y_true) ** 2)

    def rmse(self, y_true, y_pred):
        """sqrt((pred - true)^2)"""
        return torch.sqrt(self.mse(y_true, y_pred))

    def mape(self, y_true, y_pred):
        """|(pred - true) / true| * 100"""
        return torch.mean(torch.abs((y_pred - y_true) / (y_true + self.eps))) * 100

    def evaluate(self, y_true, y_pred):
        """
            y_true: 1
            y_pred: (N, 1)
        """
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true, dtype=torch.float32).to(y_pred)
        
        if y_pred.numel() > 1:
            y_pred_est = y_pred.mean()
        else:
            y_pred_est = y_pred.squeeze()
            
        if y_true.numel() > 1:
            y_true = y_true.mean()
        else:
            y_true = y_true.squeeze()
            
        # y_pred = y_pred.squeeze()
        
        # estimation error
        metrics = {
            "Relative Error": self.relative_error(y_true, y_pred_est).item(),
            "Estimation MAE": self.mae(y_true, y_pred_est).item(),     
            "Estimation MSE": self.mse(y_true, y_pred_est).item(),     
            "Estimation RMSE": self.rmse(y_true, y_pred_est).item(),  
            "MAPE": self.mape(y_true, y_pred_est).item()
        }
        
        # sample-level error
        if y_pred.numel() > 1:
            pred_std = y_pred.std().item()
            pred_var = y_pred.var().item()
            sample_mae = torch.abs(y_pred - y_true).mean().item() 
            sample_mse = torch.mean((y_pred - y_true) ** 2).item()  
            
            metrics.update({
                "Prediction Std": pred_std,
                "Prediction Var": pred_var,
                "Sample MAE": sample_mae,   
                "Sample MSE": sample_mse,    
                "Sample RMSE": torch.sqrt(torch.tensor(sample_mse)).item(),
            })
        return metrics