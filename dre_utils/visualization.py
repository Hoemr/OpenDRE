import os
import math
import pickle
import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional, Union, List
import seaborn as sns
sns.set_context('poster')
sns.set_style('white')
import torch
# plt.rcParams['pdf.fonttype'] = 42  # 42 represents Type 1 font

LOW = -5
HIGH = 5
CMAP = "viridis"   # viridis, coolwarm, plasma, inferno (you can also try 'Greens', 'Oranges', 'YlGnBu')

def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def save_trajectory(model, data_samples, savedir, ntimes=101, memory=0.01, device='cpu'):
    model.eval()

    #  Sample from prior
    z_samples = torch.randn(2000, 2).to(device)

    # sample from a grid
    npts = 800
    side = np.linspace(LOW, HIGH, npts)
    xx, yy = np.meshgrid(side, side)
    xx = torch.from_numpy(xx).type(torch.float32).to(device)
    yy = torch.from_numpy(yy).type(torch.float32).to(device)
    z_grid = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], 1)

    with torch.no_grad():
        # We expect the model is a chain of CNF layers wrapped in a SequentialFlow container.
        logp_samples = torch.sum(standard_normal_logprob(z_samples), 1, keepdim=True)
        logp_grid = torch.sum(standard_normal_logprob(z_grid), 1, keepdim=True)
        t = 0
        # for cnf in model.chain:
        cnf = model
        end_time = (cnf.sqrt_end_time * cnf.sqrt_end_time)
        integration_times = torch.linspace(0, end_time, ntimes)

        z_traj, _ = cnf(z_samples, logp_samples, integration_times=integration_times, reverse=True)
        z_traj = z_traj.cpu().numpy()

        grid_z_traj, grid_logpz_traj = [], []
        inds = torch.arange(0, z_grid.shape[0]).to(torch.int64)
        for ii in torch.split(inds, int(z_grid.shape[0] * memory)):
            _grid_z_traj, _grid_logpz_traj = cnf(
                z_grid[ii], logp_grid[ii], integration_times=integration_times, reverse=True
            )
            _grid_z_traj, _grid_logpz_traj = _grid_z_traj.cpu().numpy(), _grid_logpz_traj.cpu().numpy()
            grid_z_traj.append(_grid_z_traj)
            grid_logpz_traj.append(_grid_logpz_traj)
        grid_z_traj = np.concatenate(grid_z_traj, axis=1)
        grid_logpz_traj = np.concatenate(grid_logpz_traj, axis=1)

        plt.figure(figsize=(8, 8))
        for _ in range(z_traj.shape[0]):

            plt.clf()

            # plot target potential function
            ax = plt.subplot(2, 2, 1, aspect="equal")

            ax.hist2d(data_samples[:, 0], data_samples[:, 1], range=[[LOW, HIGH], [LOW, HIGH]], bins=200)
            ax.invert_yaxis()
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.set_title("Target", fontsize=32)

            # plot the density
            ax = plt.subplot(2, 2, 2, aspect="equal")

            z, logqz = grid_z_traj[t], grid_logpz_traj[t]

            xx = z[:, 0].reshape(npts, npts)
            yy = z[:, 1].reshape(npts, npts)
            qz = np.exp(logqz).reshape(npts, npts)
            
            x_edges = np.linspace(np.min(xx) - (xx[1, 0] - xx[0, 0]) / 2,
                      np.max(xx) + (xx[1, 0] - xx[0, 0]) / 2,
                      npts + 1)
            y_edges = np.linspace(np.min(yy) - (yy[0, 1] - yy[0, 0]) / 2,
                      np.max(yy) + (yy[0, 1] - yy[0, 0]) / 2,
                      npts + 1)

            # plt.pcolormesh(xx, yy, qz)
            plt.pcolormesh(x_edges, y_edges, qz, shading='auto')
            ax.set_xlim(LOW, HIGH)
            ax.set_ylim(LOW, HIGH)
            cmap = matplotlib.cm.get_cmap(None)
            ax.set_facecolor(cmap(0.))
            ax.invert_yaxis()
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.set_title("Density", fontsize=32)

            # plot the samples
            ax = plt.subplot(2, 2, 3, aspect="equal")

            zk = z_traj[t]
            ax.hist2d(zk[:, 0], zk[:, 1], range=[[LOW, HIGH], [LOW, HIGH]], bins=200)
            ax.invert_yaxis()
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.set_title("Samples", fontsize=32)

            # plot vector field
            ax = plt.subplot(2, 2, 4, aspect="equal")

            K = 13j
            y, x = np.mgrid[LOW:HIGH:K, LOW:HIGH:K]
            K = int(K.imag)
            zs = torch.from_numpy(np.stack([x, y], -1).reshape(K * K, 2)).to(device, torch.float32)
            logps = torch.zeros(zs.shape[0], 1).to(device, torch.float32)
            dydt = cnf.odefunc(integration_times[t], (zs, logps))[0]
            dydt = -dydt.cpu().detach().numpy()
            dydt = dydt.reshape(K, K, 2)

            logmag = 2 * np.log(np.hypot(dydt[:, :, 0], dydt[:, :, 1]))
            ax.quiver(
                x, y, dydt[:, :, 0], dydt[:, :, 1],
                np.exp(logmag), cmap="coolwarm", scale=20., width=0.015, pivot="mid"
            )
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.axis("off")
            ax.set_title("Vector Field", fontsize=32)

            makedirs(savedir)
            plt.savefig(os.path.join(savedir, f"viz-{t:05d}.jpg"))
            t += 1

def trajectory_to_video(savedir):
    import subprocess
    bashCommand = 'ffmpeg -y -i {} {}'.format(os.path.join(savedir, 'viz-%05d.jpg'), os.path.join(savedir, 'traj.mp4'))
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

def plt_likelihood_dre(args, model, likelihood_fn, npts=100, memory=100, savefig=None, epoch=1, device="cpu", steps=100):
    """Visualize the likelihood learned by a density-ratio estimator (DRE)."""
    model.eval()
    model = model.to(device)
    joint = args.joint

    # 1. Create a 2-D grid in PyTorch
    side = torch.linspace(LOW, HIGH, npts, device=device)
    yy, xx = torch.meshgrid(side, side, indexing='ij')   # (npts, npts)
    x = torch.stack([xx.ravel(), yy.ravel()], dim=1)     # (npts*npts, 2)

    # 2. Evaluate log-likelihood in mini-batches
    est_logpx, NFE = [], []
    indices = torch.arange(x.size(0), device=device)
    for idx in torch.split(indices, memory ** 2):
        logp, nfe = likelihood_fn(model, x[idx], joint=joint, steps=steps)
        est_logpx.append(logp)
        NFE.append(nfe)

    est_logpx = torch.cat(est_logpx, 0)        # (npts*npts,)
    print(est_logpx.mean().item(), est_logpx.max().item(), est_logpx.min().item())
    est_px = est_logpx.exp().reshape(npts, npts).cpu().numpy()

    mean_nfe = float(torch.tensor(NFE).float().mean())

    # 3. Plot the likelihood surface
    fig = plt.figure(figsize=(3.0, 3.0))
    ax = plt.gca()
    ax.imshow(est_px, cmap=CMAP, aspect='auto')
    # cs = plt.contourf(xx, yy, est_px, cmap=CMAP, levels=20)
    # fig.colorbar(im, ax=ax)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    if savefig is not None:
        plt.savefig(f"{savefig}/logp_{epoch}.png", bbox_inches='tight', dpi=300, pad_inches=0)
        plt.close()
    else:
        plt.show()

    return mean_nfe

def plt_density_ratio(args, model, density_ratio_fn, npts=100, memory=100, savefig=None, epoch=1, device=None, steps=100):
    """
    Plots the density ratio with a clean, light-style suitable for top-tier conferences.
    Uses white background, soft color map, and subtle contour lines.
    """
    model.eval()
    model = model.to(device)
    joint = args.joint

    with torch.no_grad():
        side = torch.linspace(LOW, HIGH, npts, device=device)
        yy, xx = torch.meshgrid(side, side, indexing='ij')
        x = torch.stack([xx.ravel(), yy.ravel()], dim=1)

        est_logr, NFE = [], []
        inds = torch.arange(x.size(0), device=device)
        for ii in torch.split(inds, int(memory**2)):
            est_logr_, NFE_ = density_ratio_fn(model, x[ii], joint=joint, steps=steps)
            est_logr.append(est_logr_)
            NFE.append(NFE_)

        est_logr = torch.cat(est_logr, 0).reshape(npts, npts).cpu().numpy()
        NFE = float(torch.tensor(NFE).float().mean())

        fig, ax = plt.subplots(figsize=(3., 3.))

        im = ax.imshow(est_logr, cmap=CMAP, aspect='equal')

        # Contour lines: use gray or same colormap's darker shade, thin and semi-transparent
        levels_contour = 10
        ax.contour(est_logr, levels=levels_contour, colors='gray', alpha=0.4, linewidths=0.4)

        # Optional: Add a colorbar on the right
        # cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
        # cbar.ax.tick_params(labelsize=8)  # Small font for colorbar ticks

        # Remove ticks and labels for minimalism
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False)

        # Set aspect to equal
        ax.set_aspect('equal', adjustable='box')

        # Ensure white background (important!)
        ax.set_facecolor('white')  # Explicitly set facecolor
        fig.patch.set_facecolor('white')  # Also set figure background

        # Save with high quality
        plt.tight_layout()
        if savefig is not None:
            plt.savefig(f"{savefig}/logr_{epoch}.png", bbox_inches='tight', dpi=300, pad_inches=0)
            plt.close(fig)
        else:
            plt.show()

    return NFE

def plt_standard_normal_likelihood(args, npts=100, savefig=None, epoch=1, device="cpu"):
    """
    Visualize the likelihood of a 2D standard normal distribution (mean=0, cov=I).
    Uses the same plotting style as plt_likelihood_dre.
    """
    # 1. Create a 2-D grid in PyTorch
    side = torch.linspace(LOW, HIGH, npts, device=device)
    yy, xx = torch.meshgrid(side, side, indexing='ij')   # (npts, npts)
    x = torch.stack([xx.ravel(), yy.ravel()], dim=1)     # (npts*npts, 2), shape: (npts*npts, 2)

    # 2. Compute log-likelihood for 2D Standard Normal: log p(x) = -0.5 * (x^2 + y^2) - log(2*pi)
    # x is of shape (npts*npts, 2), where x[:, 0] is x-coord, x[:, 1] is y-coord
    log_likelihood = -0.5 * (x[:, 0]**2 + x[:, 1]**2) - np.log(2 * np.pi)

    # 3. Reshape for plotting
    est_logpx = log_likelihood.reshape(npts, npts).cpu().numpy() # Shape: (npts, npts)
    est_px = np.exp(est_logpx) # Convert log-probability to probability

    # 4. Plot the likelihood surface using the same style as plt_likelihood_dre
    fig = plt.figure(figsize=(3.0, 3.0))
    ax = plt.gca()
    
    # Use the same colormap as plt_likelihood_dre (assumed to be defined globally or passed via args)
    # You might want to define CMAP = 'viridis' or another colormap here if it's not global
    ax.imshow(est_px, cmap=CMAP, aspect='auto')
    
    # Clean up axes as in the original
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    if savefig is not None:
        plt.savefig(f"{savefig}/lognorm_{epoch}.png", bbox_inches='tight', dpi=300, pad_inches=0)
        plt.close()
    else:
        plt.show()

def visualize_fdiv(args, fdiv_db, fdiv_true_db, savefig=None):

  plt.figure(figsize=(12, 5))
  plt.plot(range(0, len(fdiv_db) * args.eval_freq, args.eval_freq), fdiv_db, '-o', markersize=5, label='est. fDiv')
  plt.plot(range(0, len(fdiv_true_db) * args.eval_freq, args.eval_freq), fdiv_true_db, color='black', label='true fDiv')
  plt.legend(loc='lower right')
  sns.despine()
  plt.tight_layout()

  if savefig is not None:
    plt.savefig(savefig + "/fdiv.png", bbox_inches='tight')
    plt.close()
  else:
    plt.show()

def plt_potential_func(potential, ax, npts=100, title="$p(x)$"):
    """
    Args:
        potential: computes U(z_k) given z_k
    """
    xside = np.linspace(LOW, HIGH, npts)
    yside = np.linspace(LOW, HIGH, npts)
    xx, yy = np.meshgrid(xside, yside)
    z = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    z = torch.Tensor(z)
    u = potential(z).cpu().numpy()
    p = np.exp(-u).reshape(npts, npts)

    plt.pcolormesh(xx, yy, p)
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)


def plt_flow(prior_logdensity, transform, ax, npts=100, title="$q(x)$", device="cpu"):
    """
    Args:
        transform: computes z_k and log(q_k) given z_0
    """
    side = np.linspace(LOW, HIGH, npts)
    xx, yy = np.meshgrid(side, side)
    z = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    z = torch.tensor(z, requires_grad=True).type(torch.float32).to(device)
    logqz = prior_logdensity(z)
    logqz = torch.sum(logqz, dim=1)[:, None]
    z, logqz = transform(z, logqz)
    logqz = torch.sum(logqz, dim=1)[:, None]

    xx = z[:, 0].cpu().numpy().reshape(npts, npts)
    yy = z[:, 1].cpu().numpy().reshape(npts, npts)
    qz = np.exp(logqz.cpu().numpy()).reshape(npts, npts)

    plt.pcolormesh(xx, yy, qz)
    ax.set_xlim(LOW, HIGH)
    ax.set_ylim(LOW, HIGH)
    cmap = matplotlib.cm.get_cmap(None)
    ax.set_facecolor(cmap(0.))
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)


def plt_flow_density(prior_logdensity, inverse_transform, ax, npts=100, memory=100, title="$q(x)$", device="cpu"):
    side = np.linspace(LOW, HIGH, npts)
    xx, yy = np.meshgrid(side, side)
    x = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    x = torch.from_numpy(x).type(torch.float32).to(device)
    zeros = torch.zeros(x.shape[0], 1).to(x)

    z, delta_logp = [], []
    inds = torch.arange(0, x.shape[0]).to(torch.int64)
    for ii in torch.split(inds, int(memory**2)):
        z_, delta_logp_ = inverse_transform(x[ii], zeros[ii])    # 返回model(x, logpx, reverse=False)
        z.append(z_)
        delta_logp.append(delta_logp_)
    z = torch.cat(z, 0)
    delta_logp = torch.cat(delta_logp, 0)

    logpz = prior_logdensity(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
    logpx = logpz - delta_logp

    px = np.exp(logpx.cpu().numpy()).reshape(npts, npts)

    ax.imshow(px, cmap=CMAP)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)


def plt_flow_samples(prior_sample, transform, ax, npts=100, memory=100, title=r"$x \sim q(x)$", device="cpu"):
    # 绘制生成出来的数据样本, transform为采样函数，输入为Prior样本
    z = prior_sample(npts * npts, 2).type(torch.float32).to(device)
    zk = []
    inds = torch.arange(0, z.shape[0]).to(torch.int64)
    for ii in torch.split(inds, int(memory**2)):   # batch思想
        zk.append(transform(z[ii]))
    zk = torch.cat(zk, 0).cpu().numpy()
    ax.hist2d(zk[:, 0], zk[:, 1], range=[[LOW, HIGH], [LOW, HIGH]], bins=npts)
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)


def plt_samples(samples, ax, npts=100, title=r"$x \sim p(x)$"):
    ax.hist2d(samples[:, 0], samples[:, 1], range=[[LOW, HIGH], [LOW, HIGH]], bins=npts, cmap=CMAP)
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)


def visualize_transform(
    potential_or_samples, prior_sample, prior_density, transform=None, inverse_transform=None, samples=True, npts=100,
    memory=100, device="cpu"
):
    """Produces visualization for the model density and samples from the model."""
    plt.clf()
    ax = plt.subplot(1, 3, 1, aspect="equal")    # 实际样本
    if samples:
        pass
        # plt_samples(potential_or_samples, ax, npts=npts)
    else:
        plt_potential_func(potential_or_samples, ax, npts=npts)

    ax = plt.subplot(1, 3, 2, aspect="equal")     # 估计的似然
    if inverse_transform is None:
        plt_flow(prior_density, transform, ax, npts=npts, device=device)
    else:
        plt_flow_density(prior_density, inverse_transform, ax, npts=npts, memory=memory, device=device)

    ax = plt.subplot(1, 3, 3, aspect="equal")    # 生成的样本
    if transform is not None:
        plt_flow_samples(prior_sample, transform, ax, npts=npts, memory=memory, device=device)


# In[] ---------------------- Illustration for secant alignment identity (ISA-DRE) -----------------------------
def visualize_integral_trajectories(u_0_t, u_t_t, t_values, num_sample_trajectories=10, 
                                   save_path=None, cmap='plasma', is_integral=False):
    """
    Visualize the distribution of integral trajectories
    
    Args:
        u_0_t: u(x, 0, t) values, shape [batch_size, num_t]
        u_t_t: u(x, t, t) values, shape [batch_size, num_t]
        t_values: t values array, shape [num_t]
        num_sample_trajectories: number of sample trajectories to overlay
        save_path: image save path (None means no save)
        cmap: color map
    """
    # plt.rcParams.update({
    #     'font.size': 12,               # Default font size (affects xlabel, ylabel, etc.)
    #     'axes.labelsize': 12,          # Axis labels
    #     'axes.titlesize': 14,          # Title
    #     'xtick.labelsize': 12,         # X-axis tick labels ← Key: smaller than xlabel
    #     'ytick.labelsize': 12,         # Y-axis tick labels
    #     'legend.fontsize': 10,         # Legend
    #     'pdf.fonttype': 42,
    # })
    
    batch_size, num_t = u_0_t.shape
    
    # Prepare data - flatten all points
    t_flat = np.tile(t_values, batch_size)  # Repeat t values batch_size times
    u_0_t_flat = u_0_t.reshape(-1)         # Flatten all u(x,0,t) values
    u_t_t_flat = u_t_t.reshape(-1)         # Flatten all u(x,t,t) values
    
    # Overlay random sample trajectories
    sample_indices = np.random.choice(batch_size, num_sample_trajectories, replace=False)
    
    # Create figure
    if is_integral:
        figsize=(2.4 * 4, 4)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
        
        # ---------------------------
        # A. u(x, 0, t) trajectory distribution
        # ---------------------------
        hb1 = ax1.hexbin(t_flat, u_0_t_flat, 
                        gridsize=50, cmap=cmap, 
                        mincnt=1, bins='log')
        fig.colorbar(hb1, ax=ax1, label="Frequency")
        # ax1.set_title('$u(\mathbf{x}, 0, t)$ Trajectories', fontsize=14)
        ax1.set_xlabel('Time $t$', fontsize=8)
        ax1.set_ylabel('The secant function $u(x, 0, t)$', fontsize=8)
        ax1.grid(alpha=0.3)
        
        for idx in sample_indices:
            ax1.plot(t_values, u_0_t[idx], 'orange', lw=1.1, alpha=0.7)
    else:
        figsize=(8, 6)
        fig, ax2 = plt.subplots(1, 1, figsize=figsize)
        # ax2 = plt.figure(figsize=figsize)
    
    # ---------------------------
    # B. u(x, t, t) trajectory distribution
    # ---------------------------
    hb2 = ax2.hexbin(t_flat, u_t_t_flat, gridsize=50, cmap=cmap, mincnt=1, bins='log')
    # fig.colorbar(hb2, ax=ax2, label="Frequency")
    ax2.set_xlabel('Timestep $t$', fontsize=12)
    ax2.set_ylabel('Time score', fontsize=12)
    # ax2.tick_params(labelsize=12)
    ax2.grid(alpha=0.3)
    
    # Overlay the same random sample trajectories
    for idx in sample_indices:
        ax2.plot(t_values, u_t_t[idx], 'orange', lw=1.1, alpha=0.7)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0)
    else:
        plt.show()
    plt.close()

def get_integral_values(model, x_batch, num_t=100, method='fast', is_integral=False):
    """
    Calculate the values of u(x, 0, t) and u(x, t, t)
    Args:
        x_batch: input data points [batch_size, dim]
        num_t: number of t sampling points
        method: which method to use ('fast' or 'integral')
    Returns:
        u_0_t: u(x, 0, t) with shape [batch_size, num_t]
        u_t_t: u(x, t, t) with shape [batch_size, num_t]
        t_values: time sampling points [num_t]
    """
    device = x_batch.device
    batch_size = x_batch.size(0)
    t_values = torch.linspace(0, 1, num_t, device=device)

    if method == 'integral':
        # === Method 1: Numerical integration approach ===
        u_0_t = torch.zeros(batch_size, num_t, device=device)
        u_t_t = torch.zeros(batch_size, num_t, device=device)

        # Calculate u(x, t, t)
        for i, t in enumerate(t_values):
            t_tensor = t.repeat(batch_size).view(batch_size, 1)
            a_t = t.repeat(batch_size).view(batch_size, 1)
            with torch.no_grad():
                u_t_t_result = model(t_tensor, x_batch, cond=a_t)[-1]
                u_t_t[:, i] = u_t_t_result.squeeze()

        # Calculate u(x, 0, t) - numerical integration
        num_integration_points = num_t * 2
        integration_t = torch.linspace(0, 1, num_integration_points, device=device)

        for i, t_end in enumerate(t_values):
            if is_integral:
                if t_end == 0:
                    u_0_t[:, i] = u_t_t[:, i]
                    continue

                integral_sum = torch.zeros(batch_size, device=device)
                valid_t = integration_t[integration_t <= t_end]

                if len(valid_t) == 0:
                    continue

                for j in range(len(valid_t) - 1):
                    t_start = valid_t[j]
                    t_end_segment = valid_t[j + 1]
                    segment_width = t_end_segment - t_start
                    t_mid = (t_start + t_end_segment) / 2
                    t_mid_tensor = t_mid.repeat(batch_size).view(batch_size, 1)

                    with torch.no_grad():
                        s_t_result = model(t_mid_tensor, x_batch, cond=t_mid_tensor)[1]
                        s_t_val = s_t_result.squeeze()

                    integral_sum += s_t_val * segment_width

                u_0_t[:, i] = integral_sum / t_end
            else:
                u_0_t = u_t_t

    elif method == 'fast':
        # === Method 2: Direct model call approach ===
        u_0_t_list = []
        u_t_t_list = []

        for t in t_values:
            t_tensor = t.repeat(batch_size).view(batch_size, 1)
            a_0 = torch.zeros((batch_size, 1), device=device) + 1e-5  # Avoid division by zero
            a_t = t.repeat(batch_size).view(batch_size, 1)

            with torch.no_grad():
                if is_integral:
                    u_0_t = model(t_tensor, x_batch, cond=a_0)[-1]
                    u_t_t = model(t_tensor, x_batch, cond=a_t)[-1]
                else:
                    u_t_t = model(t_tensor, x_batch)[-1]
                    u_0_t = u_t_t

            u_0_t_list.append(u_0_t)
            u_t_t_list.append(u_t_t)

        u_0_t = torch.stack(u_0_t_list, dim=1).squeeze(-1)
        u_t_t = torch.stack(u_t_t_list, dim=1).squeeze(-1)

    else:
        raise ValueError("Invalid method. Choose 'integral' or 'fast'.")

    return u_0_t, u_t_t, t_values
    
def plot_integral_distributions(model, batch, sample_path, is_integral=False, path_type="VP", epoch=None):
    """
    Calculate and visualize the integral trajectory distributions for the entire batch
    
    Args:
        batch: input data batch
        epoch: current epoch (for filename)
        is_integral: bool value, only SAI is easy to derive integral, so commonly "is_integral=args.SAI"
    """
    # Get integral values
    u_0_t, u_t_t, t_values = get_integral_values(model, batch, method='fast', is_integral=is_integral)
    u_0_t = u_0_t.cpu().numpy()
    u_t_t = u_t_t.cpu().numpy()
    t_values = t_values.cpu().numpy()
    
    save_path = os.path.join(sample_path, f"integral_trajectories_{epoch}.pdf") if epoch else None
    visualize_integral_trajectories(
        u_0_t, u_t_t, t_values,
        num_sample_trajectories=min(10, u_0_t.shape[0]),  # Display at most 20 sample trajectories
        save_path=save_path,
        cmap='viridis',  # Can be changed to 'plasma' or other color maps
        is_integral=is_integral
    )
    
# In[] Illustration for path function
def visualize_path_function(path_model, sample_path="./", path_type="mvp", constraint_type="spherical"):
    methods = ['linear', 'VP', 'cosine', 'follmer', 'trigonometric']   
    colors = {'linear': 'green', 'VP': 'black', 'cosine': 'orange', 'follmer': 'pink', 'trigonometric': 'purple'}
    labels = {'linear': 'Linear', 'VP': 'VP', 'cosine': 'Cosine', 'follmer': 'Follmer', 'trigonometric': 'Trigonometric'}
    
    # get data    
    if path_type == "mvp":
        prior_traj = path_model.bayesian_path.sample_trajectories(mode="prior")
        post_traj = path_model.bayesian_path.sample_trajectories(mode="posterior")
        t_grid = path_model.bayesian_path.grid
        
        pickle_save_path = os.path.join(sample_path, f"trajectories.p")
        with open(pickle_save_path, 'wb') as f:
            pickle.dump({
                'prior': {k: v.detach().cpu() for k, v in prior_traj.items()},
                'posterior': {k: v.detach().cpu() for k, v in post_traj.items()},
                't_grid': t_grid.cpu()
            }, f)
    else:
        eps = 1e-5
        t_grid = torch.linspace(eps, 1-eps, 200).view(-1, 1)
    
    method_data = {}
    for m in methods:
        alpha, beta = path_model.get_alpha_t_beta_t(t_grid, path_type=m)
        method_data[m] = {
            'alpha': alpha.cpu(),
            'beta': beta.cpu()
        }
        
    # plot figure
    for name in ['alpha', 'beta']:
        plt.figure(figsize=(8, 6))
        
        for m in methods:
            plt.plot(t_grid.cpu(), method_data[m][name].view(t_grid.shape), color=colors[m], lw=2, linestyle='-', label=labels[m])
        
        # mvp
        if path_type == "mvp":
            alpha_post = post_traj[name].detach().cpu()
            plt.plot(t_grid.cpu(), alpha_post.view(t_grid.shape), 'r-', lw=2.5, label='MVP (Ours)')
        
        ylabel = {"alpha": '$\\alpha(t)$', "beta": '$\\beta(t)$', "gamma": '$\\gamma(t)$'}[name]
        plt.title(f'Path Function Comparison ({ylabel})', fontsize=16)
        plt.xlabel('Time $t$', fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        save_path = os.path.join(sample_path, f"path_comparison_{name}.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

def visualize_variance_profiles(path_model, sample_path="./", path_type="VP"):
    """
    Generates and saves the plot comparing variance profiles of different schedules.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if path_type == "mvp":
        prior_traj = path_model.bayesian_path.sample_trajectories(mode="prior", return_derivs=True)
        post_traj = path_model.bayesian_path.sample_trajectories(mode="posterior", return_derivs=True)
        t_grid = path_model.bayesian_path.grid
        
    else:
        eps = 1e-5  # Time grid, clipped to avoid division by zero in some derivatives (e.g., Follmer)
        t_grid = torch.linspace(eps, 1-eps, 500).view(-1, 1)
    
    methods = ['linear', 'VP', 'cosine', 'follmer', 'trigonometric']
    colors = {'linear': 'green', 'VP': 'black', 'cosine': 'orange', 'follmer': 'pink', 'trigonometric': 'purple'}
    labels = {'linear': 'Linear', 'VP': 'VP', 'cosine': 'Cosine', 'follmer': 'Follmer', 'trigonometric': 'Trigonometric'}
    
    # Baseline methods
    for method in methods:
        coefficients_fn, derivatives_fn = path_model.get_coefficients_and_derivatives_fns(path_type=method)
        alpha, beta = coefficients_fn(t_grid)
        dot_alpha, dot_beta = derivatives_fn(t_grid, alpha, beta)

        variance = path_model.score_var_fn(t_grid, alpha_t=alpha, beta_t=beta, d_alpha_dt=dot_alpha, d_beta_dt=dot_beta)[0]
        ax.plot(t_grid.detach().cpu(), variance.view(t_grid.shape).detach().cpu().log(), label=labels[method], color=colors[method], linewidth=2.5)
        
    # mvp method
    if path_type == "mvp":
        alpha = post_traj["alpha"]
        beta = post_traj["beta"]
        dot_alpha = post_traj["dot_alpha"]
        dot_beta = post_traj["dot_beta"]
        variance = path_model.score_var_fn(t_grid, alpha_t=alpha, beta_t=beta, d_alpha_dt=dot_alpha, d_beta_dt=dot_beta)[0]
        ax.plot(t_grid.detach().cpu(), variance.view(t_grid.shape).detach().cpu().log(), 'r-', lw=2.5, label="MVP (Ours)")

    # --- Plot Customization ---
    # ax.set_yscale('log')
    # ax.set_title('Path-Dependent Variance of Different Interpolation Schedules', fontsize=18, pad=15)
    ax.set_xlabel('Time $t$', fontsize=14) 
    ax.set_ylabel('Log Path Variance', fontsize=14)
    ax.legend(fontsize=12, loc='best')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(-0.05, 1.05)
    ax.axvspan(0.9, 1.02, color='red', alpha=0.1, label='Unstable Region')
    ax.text(0.95, variance.detach().cpu().log().mean(), 'Potential Instability', rotation=90, verticalalignment='bottom', color='red', alpha=0.7, fontsize=14)

    plt.tight_layout()
    save_path = os.path.join(sample_path, f"variance_comparison_{path_type}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def visualize_integrated_variance_profiles(path_model, sample_path="./", path_type="VP"):
    """
    Generates and saves the plot comparing integrated variance profiles of different schedules.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if path_type == "mvp":
        prior_traj = path_model.bayesian_path.sample_trajectories(mode="prior", return_derivs=True)
        post_traj = path_model.bayesian_path.sample_trajectories(mode="posterior", return_derivs=True)
        t_grid = path_model.bayesian_path.grid
    else:
        eps = 1e-5
        t_grid = torch.linspace(eps, 1 - eps, 200).view(-1, 1)

    methods = ['linear', 'VP', 'cosine', 'follmer', 'trigonometric']
    colors = {'linear': 'green', 'VP': 'black', 'cosine': 'orange', 'follmer': 'pink', 'trigonometric': 'purple'}
    labels = {'linear': 'Linear', 'VP': 'VP', 'cosine': 'Cosine', 'follmer': 'Follmer', 'trigonometric': 'Trigonometric'}

    cutoff = 3
    # Baseline methods
    for method in methods:
        coefficients_fn, derivatives_fn = path_model.get_coefficients_and_derivatives_fns(path_type=method)
        alpha, beta = coefficients_fn(t_grid)
        dot_alpha, dot_beta = derivatives_fn(t_grid, alpha, beta)

        variance = path_model.score_var_fn(t_grid, alpha_t=alpha, beta_t=beta, d_alpha_dt=dot_alpha, d_beta_dt=dot_beta)[0]
        variance = variance.view(t_grid.shape).detach().cpu()

        # Compute cumulative integral using trapezoidal rule
        t_tensor = t_grid.squeeze().detach().cpu()
        var_tensor = variance.squeeze()
        
        # Calculate cumulative integral using torch.trapz
        integrated_variance = torch.zeros_like(var_tensor)
        for i in range(1, len(t_tensor)):
            integrated_variance[i] = torch.trapz(var_tensor[:i+1], t_tensor[:i+1])

        # Take log of integrated variance (add small epsilon to avoid log(0))
        log_integrated_variance = torch.log(integrated_variance + 1e-12)

        ax.plot(t_tensor.numpy()[cutoff:-cutoff], log_integrated_variance.numpy()[cutoff:-cutoff], 
                label=labels[method], color=colors[method], linewidth=2.5)

    # mvp method
    if path_type == "mvp":
        alpha = post_traj["alpha"]
        beta = post_traj["beta"]
        dot_alpha = post_traj["dot_alpha"]
        dot_beta = post_traj["dot_beta"]
        variance = path_model.score_var_fn(t_grid, alpha_t=alpha, beta_t=beta, d_alpha_dt=dot_alpha, d_beta_dt=dot_beta)[0]
        variance = variance.view(t_grid.shape).detach().cpu()

        t_tensor = t_grid.squeeze().detach().cpu()
        var_tensor = variance.squeeze()
        
        # Calculate cumulative integral using torch.trapz
        integrated_variance = torch.zeros_like(var_tensor)
        for i in range(1, len(t_tensor)):
            integrated_variance[i] = torch.trapz(var_tensor[:i+1], t_tensor[:i+1])

        # Take log of integrated variance (add small epsilon to avoid log(0))
        log_integrated_variance = torch.log(integrated_variance + 1e-12)

        ax.plot(t_tensor.numpy()[cutoff:-cutoff], log_integrated_variance.numpy()[cutoff:-cutoff], 
                'r-', lw=2.5, label="MVP (Ours)")

    # --- Plot Customization ---
    ax.set_xlabel('Time $t$', fontsize=14)
    ax.set_ylabel('Log Path Variance', fontsize=14)
    ax.legend(fontsize=12, loc='upper left')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(-0.05, 1.05)
    ax.axvspan(0.9, 1.02, color='red', alpha=0.1, label='Unstable Region')
    ax.text(0.95, log_integrated_variance.min(), 'Potential Instability', rotation=90, verticalalignment='bottom',
            color='red', alpha=0.7, fontsize=14)

    plt.tight_layout()
    save_path = os.path.join(sample_path, f"integrated_variance_comparison_{path_type}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

# In[] Change Point Detection Statistic
def visualize_cpd_stat(
    cp_stats: Union[np.ndarray, List[np.ndarray]],
    t_values: Optional[np.ndarray] = None,
    true_change_points: Optional[List[int]] = None,
    window_length: int = 64,
    save_path: Optional[str] = None,
    title: str = "Change Point Detection Statistic",
    xlabel: str = "Time",
    ylabel: str = "CPD Statistic $|\\bar{r}_L - \\bar{r}_R|$",
    figsize: tuple = (8, 5),
    fill_alpha: float = 0.3,
    line_width: float = 1.8,
    show_individual: bool = False,
    individual_alpha: float = 0.4
):
    """
    Visualize change point detection (CPD) statistic with uncertainty band.
    
    Args:
        cp_stats: 
            - If np.ndarray of shape (T,): single run
            - If List[np.ndarray] or np.ndarray of shape (N, T): multiple runs for error band
        t_values: Optional time axis (length T). If None, use 0,1,...,T-1.
        true_change_points: Optional list of ground truth change point indices to mark.
        window_length: Sliding window length (used to trim warm-up region).
        save_path: Path to save figure (e.g., 'cpd_stat.png'). If None, plt.show().
        title, xlabel, ylabel: Plot labels.
        cmap: Color for main curve (and individual traces if shown).
        figsize: Figure size.
        fill_alpha: Transparency of error band.
        line_width: Width of mean curve.
        show_individual: Whether to overlay individual runs (only if multiple runs provided).
        individual_alpha: Transparency of individual runs.
    """
    # Handle input format
    if isinstance(cp_stats, list):
        cp_stats = np.array(cp_stats)  # shape (N, T)
    if cp_stats.ndim == 1:
        cp_stats = cp_stats[None, :]   # shape (1, T)

    N, T = cp_stats.shape
    if t_values is None:
        t_values = np.arange(T)

    # Trim warm-up region (before 2*window_length is unreliable)
    warmup = window_length  # or window_length // 2 depending on your stat design
    valid_mask = t_values >= warmup
    t_plot = t_values[valid_mask]
    cp_plot = cp_stats[:, valid_mask]

    # Compute mean and std
    mean_stat = cp_plot.mean(axis=0)
    std_stat = cp_plot.std(axis=0)

    # Plot
    plt.figure(figsize=figsize)
    
    # # Error band
    # plt.fill_between(t_plot, mean_stat - std_stat, mean_stat + std_stat, 
    #                  color='C0', alpha=fill_alpha, label='±1 std')
    
    # Mean curve
    plt.plot(t_plot, mean_stat, color='C0', lw=line_width, label='Mean CPD Stat')

    # Optional: overlay individual runs
    if show_individual and N > 1:
        for i in range(min(N, 10)):  # limit to 10 traces for clarity
            plt.plot(t_plot, cp_plot[i], color='C0', lw=0.8, alpha=individual_alpha)

    # Optional: mark true change points
    if true_change_points is not None:
        for cp in true_change_points:
            if cp >= warmup:
                plt.axvline(cp, color='red', linestyle='--', lw=1.5, alpha=0.8)

    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.title(title, fontsize=12)
    plt.grid(alpha=0.3)
    if true_change_points:
        plt.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()