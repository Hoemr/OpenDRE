import os
import torch
import torch.distributions as dist
from tqdm import tqdm
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np


def plot_gradient_field(energy_fn, device="cpu", save_path="gradient_field.png"):
    # Generate grid data
    x_grid = torch.linspace(-3, 3, 20, device=device, requires_grad=True).reshape(-1, 1, 1, 1)

    # Compute gradient field
    with torch.enable_grad():
        U, grad = energy_fn(x_grid)

    # Convert to NumPy arrays
    x_np = x_grid.detach().squeeze().cpu().numpy()  # [20]
    grad_np = grad.detach().squeeze().cpu().numpy()  # [20]

    # Create figure
    plt.figure(figsize=(10, 2), dpi=300)

    # Plot gradient field
    plt.quiver(
        x_np,  # X coordinates
        np.zeros_like(x_np),  # Y coordinates (all zeros)
        grad_np,  # X-component of vectors
        np.zeros_like(grad_np),  # Y-component of vectors
        scale=150,  # Arrow scaling factor
        width=0.002,  # Arrow width
        headwidth=5,  # Arrow head width
        color="blue",
    )

    # Aesthetics
    plt.title("Gradient Field Visualization", fontsize=10)
    plt.xlim(-3.5, 3.5)
    plt.axis("off")  # Turn off axes
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def plot_2d_gradient_field(
    energy_fn,
    data_shape=(1, 28, 20),
    device="cpu",
    save_path="2d_gradient_field.png",
    grid_step=3,
):
    """
    Visualize a 2D gradient field (suitable for images or field data)

    Args:
        energy_fn: Energy function that takes input x and returns (U, grad)
        data_shape: Data shape (C, H, W)
        device: Computation device
        save_path: Path to save the figure
        grid_step: Spacing between grid points (reduces density to avoid overlapping arrows)
    """
    # Generate 2D spatial grid (only spatial dimensions)
    H, W = data_shape[1], data_shape[2]
    y_grid = torch.arange(0, H, grid_step).float().to(device)  # Grid along height
    x_grid = torch.arange(0, W, grid_step).float().to(device)  # Grid along width
    grid_y, grid_x = torch.meshgrid(y_grid, x_grid, indexing="ij")  # [H_grid, W_grid]

    # Construct dummy data (assume 1 channel, initialized to 0)
    batch_size = grid_x.numel()
    x_data = torch.zeros(batch_size, *data_shape).to(device)  # [N, C, H, W]

    # Activate grid points (assuming image coordinate system with origin at top-left)
    # If data is normalized, coordinate range should be adjusted (e.g., from -1 to 1)
    x_data[:, 0, grid_y.long(), grid_x.long()] = 1.0  # Set activation at grid positions

    # Compute gradients
    with torch.enable_grad():
        _, grad = energy_fn(x_data)  # grad shape: [N, C, H, W]
        grad = grad.mean(dim=0)  # Average across batch (assuming positional independence)

    # Extract spatial gradient components (assume gradients exist along H and W)
    grad_y = grad[0, :, :].cpu().numpy()  # Gradient along height [H, W]
    grad_x = grad[0, :, :].cpu().numpy()  # Gradient along width [H, W]

    # Create figure
    plt.figure(figsize=(10, 10 * (H / W)), dpi=150)  # Maintain aspect ratio

    # Plot gradient field
    plt.quiver(
        grid_x.cpu().numpy(),  # X coordinates (width direction)
        grid_y.cpu().numpy(),  # Y coordinates (height direction)
        grad_x[::grid_step, ::grid_step],  # X-component (width-direction gradient)
        -grad_y[::grid_step, ::grid_step],  # Y-component (invert height gradient)
        scale=50,  # Arrow scaling factor
        width=0.003,  # Arrow line width
        headwidth=3,  # Arrow head width
        color="blue",
        angles="xy",  # Ensure correct arrow orientation
        scale_units="xy",  # Preserve aspect ratio
    )

    # Aesthetics
    plt.title("2D Gradient Field", fontsize=12)
    plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinate system
    plt.axis("off")
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def AIS_sampler(
    logr_fn,
    path_model,
    n,
    data_shape=(1, 28, 28),
    num_steps=100,
    step_size=0.1,
    device="cpu",
    **kwargs,
):
    with torch.no_grad():
        # logr_fn: takes input x and returns log(p_data(x) / p_noise(x))
        # Initialize samples and log-weights
        x = path_model.prior_sampling((n, *data_shape)).to(device)  # Sample from p0
        dims = list(range(1, x.ndim))
        log_weights = torch.zeros(n).to(x)
        # beta_schedule = torch.linspace(0, 1, num_steps).to(x)  # Annealing schedule
        # beta_schedule = torch.sigmoid(torch.linspace(-5, 5, num_steps).to(x))
        beta_schedule = torch.linspace(0, 1, num_steps) ** 2

        torch.cuda.empty_cache()  # Clear cache

        prev_beta = 0.0
        for beta in beta_schedule:
            delta_beta = beta - prev_beta
            log_r = logr_fn(x)
            log_weights += delta_beta * log_r  # Accumulate log r(x) directly

            # Metropolis-Hastings transition step (random-walk proposal)
            x_proposal = x + torch.randn_like(x, device=x.device) * step_size

            # Compute acceptance probability
            log_r_proposal = logr_fn(x_proposal)
            log_p0 = path_model.prior_logp(x)  # log-density of p0 (standard normal)
            log_p0_proposal = path_model.prior_logp(x_proposal)

            # Acceptance probability formula (note beta parameterization)
            log_accept = log_p0_proposal + beta * log_r_proposal - (log_p0 + beta * log_r)
            log_accept = torch.clamp(log_accept, max=0)

            # Decide whether to accept proposal
            accept = torch.rand(n).to(x) < torch.exp(log_accept)
            accept = accept.view(-1, 1, 1, 1)
            x = torch.where(accept, x_proposal, x)
            prev_beta = beta

        # Resample based on final weights
        weights = torch.exp(log_weights - torch.max(log_weights))  # Numerical stability
        weights /= weights.sum()
        indices = torch.multinomial(weights, n, replacement=True)  # With replacement
        torch.cuda.empty_cache()
    return x[indices]  # [n, C, H, W]


def HMC_sampler(
    logr_fn,
    path_model,
    n,
    data_shape=(1, 28, 28),
    num_steps=500,
    step_size=0.1,
    leapfrog_steps=30,
    device="cpu",
    **kwargs,
):
    x = path_model.prior_sampling((n, *data_shape)).to(device)
    target_accept_rate = 0.65

    def get_energy_and_grad(x_in):
        x_in = x_in.detach().requires_grad_(True)
        with torch.enable_grad():
            log_r = logr_fn(x_in)
            log_p0 = path_model.prior_logp(x_in)
            U = -(log_r + log_p0)
            # create_graph=False 极其重要，我们不需要二阶导
            grad = torch.autograd.grad(U.sum(), x_in, create_graph=False)[0]
        return U.detach(), grad.detach()

    U, grad = get_energy_and_grad(x)

    for step in tqdm(range(num_steps), desc="HMC Sampling"):
        # 1. Sample Momentum
        v = torch.randn_like(x)

        # 2. Metropolis Proposal (Leapfrog)
        x_new = x.clone()
        v_new = v.clone()
        U_new, grad_new = U, grad  # Start with current state

        # Half-step momentum update
        v_new = v_new - 0.5 * step_size * grad_new

        # Full-step position and momentum updates
        for i in range(leapfrog_steps):
            # Full step position
            x_new = x_new + step_size * v_new

            # Update gradient at new position
            U_new, grad_new = get_energy_and_grad(x_new)

            if i != leapfrog_steps - 1:
                v_new = v_new - step_size * grad_new

        # Final half-step momentum update
        v_new = v_new - 0.5 * step_size * grad_new

        # 3. Metropolis Acceptance
        # Kinetic Energies
        K_current = 0.5 * (v.flatten(1) ** 2).sum(1)
        K_new = 0.5 * (v_new.flatten(1) ** 2).sum(1)

        # Energy difference
        dH = (U_new + K_new) - (U + K_current)

        # Accept mask: u ~ U[0,1]，if log(u) < -dH, accept
        log_u = torch.log(torch.rand(n, device=device))
        accept_mask = log_u < -dH

        # Update state
        mask_indices = accept_mask.nonzero().squeeze()
        if mask_indices.numel() > 0:
            x[mask_indices] = x_new[mask_indices]
            U[mask_indices] = U_new[mask_indices]
            grad[mask_indices] = grad_new[mask_indices]

        # 4. Step Size Adaptation (Naive)
        accept_ratio = accept_mask.float().mean().item()
        if accept_ratio > target_accept_rate:
            step_size *= 1.02
        else:
            step_size *= 0.98

    return x


def AIS_sampler_HMC(
    logr_fn,
    path_model,
    n,
    data_shape=(1, 28, 28),
    num_steps=100,
    leapfrog_steps=10,
    step_size=0.1,
    device="cuda",
    **kwargs,
):
    # 1. Initialize from Prior
    x = path_model.prior_sampling((n, *data_shape)).to(device)
    log_weights = torch.zeros(n, device=device)

    # Sigmoid schedule for beta (smooth interpolation)
    beta_schedule = torch.sigmoid(torch.linspace(-5, 5, num_steps, device=device))

    # Helper: Compute U(x) and grad_U(x) efficiently
    # U(x) = - (log_p0(x) + beta * log_r(x))
    def get_potential_and_grad(x_in, beta):
        x_in = x_in.detach().requires_grad_(True)
        with torch.enable_grad():
            log_p0 = path_model.prior_logp(x_in)
            log_r = logr_fn(x_in)
            log_p = log_p0 + beta * log_r
            U = -log_p.sum()
            grad = torch.autograd.grad(U, x_in, create_graph=False)[0]
        # Return detached values to keep the main loop graph-free
        return -log_p.detach(), grad.detach()  # Return U (energy) and grad

    # Initial energy and gradient at beta=0
    # Note: At beta=0, U = -log_p0
    curr_U, curr_grad = get_potential_and_grad(x, beta_schedule[0])

    prev_beta = 0.0

    for step, beta in enumerate(tqdm(beta_schedule, desc="AIS-HMC")):
        # --- 1. Weight Update (AIS) ---
        # log w += (beta_t - beta_{t-1}) * log r(x_{t-1})
        delta_beta = beta - prev_beta
        with torch.no_grad():
            log_r = logr_fn(x)
            log_weights += delta_beta * log_r

        # --- 2. HMC Transition Kernel ---
        # Sample Momentum
        v = torch.randn_like(x)

        # Current Hamiltonian
        # H = U + K = U + 0.5 * ||v||^2
        curr_K = 0.5 * (v.flatten(1) ** 2).sum(1)
        curr_H = curr_U + curr_K

        # Initialize Proposal
        x_new = x.clone()
        v_new = v.clone()
        # We reuse curr_grad for the first half-step

        # Leapfrog Integration
        # Initial half-step momentum update
        v_new = v_new - 0.5 * step_size * curr_grad

        for i in range(leapfrog_steps):
            # Full-step position update
            x_new = x_new + step_size * v_new

            # Update gradient at new position
            # Important: Use CURRENT beta for the potential energy
            U_new, grad_new = get_potential_and_grad(x_new, beta)

            # Full-step momentum update (except last step)
            if i != leapfrog_steps - 1:
                v_new = v_new - step_size * grad_new

        # Final half-step momentum update
        v_new = v_new - 0.5 * step_size * grad_new

        # Metropolis-Hastings Acceptance
        K_new = 0.5 * (v_new.flatten(1) ** 2).sum(1)
        H_new = U_new + K_new

        # log_accept = H_old - H_new (standard MH)
        log_accept_ratio = curr_H - H_new
        # Accept if log(u) < log_accept_ratio
        accept_mask = torch.log(torch.rand(n, device=device)) < log_accept_ratio

        # Update State
        mask_indices = accept_mask.nonzero().squeeze()
        if mask_indices.numel() > 0:
            x[mask_indices] = x_new[mask_indices]
            curr_U[mask_indices] = U_new[mask_indices]
            curr_grad[mask_indices] = grad_new[mask_indices]

        prev_beta = beta

    return x


SAMPLERS = {
    "AIS": AIS_sampler,
    "AIS-HMC": AIS_sampler_HMC,
    "HMC": HMC_sampler,
}


def sampling(
    score_model,
    path_model,
    logr_fn,
    n_samples,
    data_shape,
    joint=True,
    method="AIS",
    num_steps=100,
    step_size=0.1,
    device="cpu",
    save_dir="",
    filename="",
    **kwargs,
):
    if len(data_shape) == 3:
        if "normalize_const" in kwargs:
            channels, mean, std = kwargs["normalize_const"]
        else:
            channels = data_shape[0]
            mean = 0.0
            std = 1.0

        mean = torch.tensor(mean, device=device).view(1, channels, 1, 1)
        std = torch.tensor(std, device=device).view(1, channels, 1, 1)
    else:
        mean = torch.tensor(0.0, device=device).view(1, 1)
        std = torch.tensor(1.0, device=device).view(1, 1)

    def partial_logr_fn(x):
        # return score_model.compute_weighted_sum(x)
        # return torch.from_numpy(logr_fn(score_model, x, joint)[0]).to(x)
        return (logr_fn(score_model, x, joint)[0]).to(x)

    # Sampling
    sampler = SAMPLERS[method]
    samples = sampler(
        logr_fn=partial_logr_fn,
        path_model=path_model,
        n=n_samples,
        data_shape=data_shape,
        num_steps=num_steps,
        step_size=step_size,
        device=device,
        save_dir=save_dir,
    )
    samples = samples * std + mean
    return samples

