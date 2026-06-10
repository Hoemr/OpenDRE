import torch
import numpy as np
import os
import math
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import random

def set_random_seeds(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # torch.set_default_dtype(torch.float64)

    seed_everything(seed, workers=True)

class ValidationStartCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module) -> None:
        trainer.validate(pl_module)
        
def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]
    
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def get_transforms(model):

    def sample_fn(z, logpz=None):
        if logpz is not None:
            return model(z, logpz, reverse=True)
        else:
            return model(z, reverse=True)[0]

    def density_fn(x, logpx=None):
        if logpx is not None:
            return model(x, logpx, reverse=False)
        else:
            return model(x, reverse=False)

    return sample_fn, density_fn

def compute_bits_dim(log_prob, d):
    bits_per_dim = -log_prob.mean() / (d * np.log(2))
    return bits_per_dim.item()

def compute_true_bits_dim(true_pdf, data, d):
    pdf_values = true_pdf(data.numpy())
    pdf_values = np.maximum(pdf_values, 1e-10)  
    log_prob = np.log(pdf_values)
    
    # 离散数据，比如图像数据适合这个
    # log_prob_per_dim = np.sum(log_prob) / data.nelement()
    # bits_per_dim = -(log_prob_per_dim - np.log(256)) / np.log(2)
    
    # 连续数据，比如高斯或者copula这种分布适合这个
    bits_per_dim = -log_prob.mean() / (d * np.log(2))
    return bits_per_dim

class ExponentialMovingAverage:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.ema = 0.

    def __call__(self, value):
        ema = self.alpha * self.ema + (1 - self.alpha) * value
        self.ema = ema.detach()
        return ema

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def divergence_approx(fz, z, noise_type="gauss"):  # f为z的函数，z为自变量，e为随机噪声列向量
    if noise_type == "gauss":
        e = sample_gaussian_like(z).to(z)
    elif noise_type == "rademacher":
        e = sample_rademacher_like(z).to(z)
    else:
        raise NotImplementedError
    e_dfdz = torch.autograd.grad(fz, z, e, create_graph=True)[0]  # 最后的这个[0]一定得有，不然下一行那个乘法运算没法进行
    e_dfdz_e = e_dfdz * e  # 这里e_dfdz与e的size是一样的，它们相乘是对应元素相乘
    return e_dfdz_e.sum(dim=tuple(range(1,z.ndim))).unsqueeze(-1)    # [BS, 1]

def jacobian_frobenius_divergence_approx(fz, z, noise_type="gauss"):
    """
    计算函数 f(x) 相对于 x 的雅可比矩阵的 Frobenius 范数平方的 Hutchinson 估计，
    即 tr(J^T J)，其中 J = df/dx。
    """
    if noise_type == "gauss":
        noise = torch.randn_like(fz)
    elif noise_type == "rademacher":
        noise = torch.randint(0, 2, fz.shape, device=fz.device).float() * 2 - 1
    else:
        raise NotImplementedError("noise_type 必须为 'gauss' 或 'rademacher'")
    
    # 计算向量-Jacobian 乘积：这里得到的是 J^T v
    jtv = torch.autograd.grad(outputs=fz, inputs=z, grad_outputs=noise, create_graph=True)[0]
    
    # 计算每个样本的平方范数，即 Hutchinson 估计： ||J^T v||^2
    divergence = jtv.pow(2).view(z.shape[0], -1).sum(dim=1)
    
    return divergence

def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1
    
def sample_gaussian_like(y):
    return torch.randn_like(y)

def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2

def unsqueeze(input, upscale_factor=2):
    '''
    [:, C*r^2, H, W] -> [:, C, H*r, W*r]
    '''
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels = in_channels // (upscale_factor**2)

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor

    input_view = input.contiguous().view(batch_size, out_channels, upscale_factor, upscale_factor, in_height, in_width)

    output = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    return output.view(batch_size, out_channels, out_height, out_width)

def squeeze(input, downscale_factor=2):
    '''
    [:, C, H*r, W*r] -> [:, C*r^2, H, W]
    '''
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels = in_channels * (downscale_factor**2)

    out_height = in_height // downscale_factor
    out_width = in_width // downscale_factor

    input_view = input.contiguous().view(
        batch_size, in_channels, out_height, downscale_factor, out_width, downscale_factor
    )

    output = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return output.view(batch_size, out_channels, out_height, out_width)
    
def divergence_bf(dx, y, **unused_kwargs):
    sum_diag = 0.
    for i in range(y.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()


def compute_endpoint_stat(loader, which="px", mode="scalar", stat_type="second_moment", device=None, max_batches=None):
    """Compute endpoint statistic for EDM preconditioning from train loader.

    Batch format must be (px, qx, ...). Explicitly:
    - which="px": use only the first tensor (px) from each batch.
    - which="qx": use only the second tensor (qx) from each batch.
    Do not use the whole batch or concatenate.

    stat_type: "second_moment" -> E[x^2] (default, RMS-style); "variance" -> E[(x - mu)^2].
    mode: "scalar" -> single value; "channelwise" -> per-channel (C,) for (B,C,H,W).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx = 0 if which == "px" else 1
    n = 0
    sum_x = None
    sum_sq = None
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        if isinstance(batch, (list, tuple)):
            x = batch[idx]
        else:
            x = batch
        x = x.detach().to(device, dtype=torch.float32)
        if stat_type == "second_moment":
            sq = x.pow(2)
        else:
            sq = x.pow(2)
        if mode == "scalar":
            sum_x = x.sum() if sum_x is None else sum_x + x.sum()
            sum_sq = sq.sum() if sum_sq is None else sum_sq + sq.sum()
            n += x.numel()
        else:
            # channelwise: (B, C, H, W) -> reduce (B,H,W) -> (C,)
            if x.dim() == 4:
                sum_x = x.sum(dim=(0, 2, 3)) if sum_x is None else sum_x + x.sum(dim=(0, 2, 3))
                sum_sq = sq.sum(dim=(0, 2, 3)) if sum_sq is None else sum_sq + sq.sum(dim=(0, 2, 3))
                n += x.shape[0] * x.shape[2] * x.shape[3]
            else:
                # tabular (B, D)
                sum_x = x.sum(dim=0) if sum_x is None else sum_x + x.sum(dim=0)
                sum_sq = sq.sum(dim=0) if sum_sq is None else sum_sq + sq.sum(dim=0)
                n += x.shape[0]
    if n == 0:
        return torch.ones(1, device=device)
    if stat_type == "second_moment":
        return (sum_sq / n).clamp(min=1e-8)
    # variance: E[x^2] - E[x]^2
    mean_x = sum_x / n
    return (sum_sq / n - mean_x.pow(2)).clamp(min=1e-8)

