import os
import re
import glob
import numpy as np
import pandas as pd

import torch, torchvision
import torchvision.datasets as dset
from torchvision import transforms
import torch.nn.functional as F
import torch.utils.data as torchdata

from scipy.special import digamma
from scipy.stats import gamma, beta, poisson, entropy
from scipy.linalg import block_diag
from scipy.special import betaln, psi
from sklearn import datasets as skdata
from sklearn.utils import shuffle as util_shuffle
from typing import Optional, List, Tuple
# from gensim.models import Word2Vec

from .tabular import BSDS300, POWER, GAS, HEPMASS, MINIBOONE

# In[] likelihood estimation: toy and tabular
class TreeGaussianSampler:
    def __init__(self, seed=2, scale=np.array([4., 4.]), device='cpu'):
        """
        Efficient tree-shaped Gaussian Mixture Model sampler
        
        Parameters:
            seed: Random seed
            scale: Scaling factor controlling data range
            device: Compute device ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.scale = torch.tensor(scale, dtype=torch.float32, device=self.device)
        self.components = self._build_tree(seed)
        self._build_sampling_lut()
        
    def _build_tree(self, seed):
        rnd = np.random.RandomState(seed)
        components = []
        
        def recurse(depth, pos, angle):
            if depth >= 7:
                return

            # Branch parameters
            direction = np.array([np.cos(angle), np.sin(angle)])
            dist = 0.292 * (0.8 ** depth) * (rnd.randn() * 0.2 + 1)
            thick = 0.4 * (0.8 ** depth) / dist
            size = self.scale.cpu().numpy() * dist * 0.06

            # Generate Gaussian components along the branch
            for t in np.linspace(0.07, 0.93, num=8):
                mu = (pos + direction * dist * t) * self.scale.cpu().numpy()
                outer_dir = np.outer(direction, direction)
                Sigma = (outer_dir + (np.eye(2) - outer_dir) * (thick ** 2)) * np.outer(size, size)
                weight = dist * (0.5 ** depth)
                components.append((mu, Sigma, weight))

            # Recursively generate child branches
            for sign in [1, -1]:
                new_angle = angle + sign * (0.7 ** depth) * (rnd.randn() * 0.2 + 1)
                recurse(depth + 1, pos + direction * dist, new_angle)
        
        # Generate two symmetric tree structures
        origin = np.array([0.0030, 0.0325])
        recurse(0, origin, np.pi * 0.25)  # First branch
        recurse(0, origin, np.pi * 1.25)  # Second branch
        
        return components
    
    def _build_sampling_lut(self, lut_size=64<<10):
        """Build sampling lookup table"""
        weights = np.array([c[2] for c in self.components])
        weights = weights / weights.sum()
        
        self.means = torch.tensor(
            np.array([c[0] for c in self.components]), 
            dtype=torch.float32,
            device=self.device
        )
        self.covs = torch.tensor(
            np.array([c[1] for c in self.components]), 
            dtype=torch.float32,
            device=self.device
        )
        
        # Precompute eigen decomposition for efficient sampling
        L, Q = torch.linalg.eigh(self.covs)  # covs = Q @ L @ Q^T
        self.L = L
        self.Q = Q
        
        # Construct sampling lookup table
        self.sample_lut = torch.zeros(lut_size, dtype=torch.int64, device=self.device)
        phi_ranges = (torch.cat([
            torch.zeros(1, dtype=torch.float32, device=self.device), 
            torch.tensor(weights, dtype=torch.float32, device=self.device).cumsum(0)
        ]) * lut_size + 0.5).to(torch.int32)
        
        for idx, (begin, end) in enumerate(zip(phi_ranges[:-1], phi_ranges[1:])):
            self.sample_lut[begin:end] = idx
    
    def sample(self, batch_size, sigma=0, generator=None):
        """
        Parameters:
            sigma: Noise standard deviation
            generator: Random number generator
        Returns:
            samples: Sampled points [batch_size, 2]
        """
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=self.device).broadcast_to(batch_size)
        
        # 1. Select components
        idx = self.sample_lut[torch.randint(
            len(self.sample_lut), 
            size=(batch_size,), 
            device=self.device,
            generator=generator
        )]
        
        # 2. Sample from selected components
        L = self.L[idx] + sigma[..., None] ** 2  # Add noise
        z = torch.randn((batch_size, 2), dtype=torch.float32, device=self.device, generator=generator)
        samples = torch.einsum('...ij,...j,...kj,...k->...i', 
                              self.Q[idx], L.sqrt(), self.Q[idx], z)
        
        return samples + self.means[idx]

class LikelihoodDataGenerator:
    def __init__(self, data_type='gaussian', d=2, rng=None, args=None):
        """
        Args:
            data_type (str): Type of data distribution (e.g., 'gaussian', 't', 'swissroll')
            d (int): Dimension of the data (for supported distributions)
            rng: Random number generator or seed
        """
        self.data_type = data_type
        self.d = d
        if rng is None:
            rng = np.random.RandomState(seed=args.seed)
        self.rng = rng
        
        # Initialize the generator based on data type
        self._init_generator()
    
    def _init_generator(self):
        """Initialize the appropriate generator function based on data type."""
        if self.data_type == 'swissroll':
            self._init_swissroll()
        elif self.data_type == 'circles':
            self._init_circles()
        elif self.data_type == 'rings':
            self._init_rings()
        elif self.data_type == 'moons':
            self._init_moons()
        elif self.data_type == '8gaussians':
            self._init_8gaussians()
        elif self.data_type == 'pinwheel':
            self._init_pinwheel()
        elif self.data_type == '2spirals':
            self._init_2spirals()
        elif self.data_type == 'checkerboard':
            self._init_checkerboard()
        elif self.data_type =="tree":
            self._init_tree()

        elif self.data_type in ['bsds300', 'power', 'gas', 'hepmass', 'miniboone']:
            self._init_real_dataset()
        else:
            raise ValueError(f"Unsupported distribution type: {self.data_type}")
    
    def _init_swissroll(self):
        """Initialize Swiss roll dataset generator."""
        self._sample_fn = lambda n: skdata.make_swiss_roll(
            n_samples=n, noise=1.0)[0].astype("float32")[:, [0, 2]] * 0.2
    
    def _init_circles(self):
        """Initialize circles dataset generator."""
        self._sample_fn = lambda n: skdata.make_circles(
            n_samples=n, factor=.5, noise=0.08)[0].astype("float32") * 3
    
    def _init_rings(self):
        """Initialize concentric rings dataset generator."""
        def sample_rings(n):
            n_samples4 = n_samples3 = n_samples2 = n // 4
            n_samples1 = n - n_samples4 - n_samples3 - n_samples2
            linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
            linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
            linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
            linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)
            circ4_x = np.cos(linspace4)
            circ4_y = np.sin(linspace4)
            circ3_x = np.cos(linspace4) * 0.75
            circ3_y = np.sin(linspace3) * 0.75
            circ2_x = np.cos(linspace2) * 0.5
            circ2_y = np.sin(linspace2) * 0.5
            circ1_x = np.cos(linspace1) * 0.25
            circ1_y = np.sin(linspace1) * 0.25
            X = np.vstack([
                np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
                np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
            ]).T * 3.0
            X = util_shuffle(X, random_state=self.rng)
            X = X + self.rng.normal(scale=0.08, size=X.shape)   # Add noise
            return X.astype("float32")
        
        self._sample_fn = sample_rings
    
    def _init_moons(self):
        """Initialize moons dataset generator."""
        self._sample_fn = lambda n: (
            skdata.make_moons(n_samples=n, noise=0.1)[0]
            .astype("float32") * 2 + np.array([-1, -0.2]))
    
    def _init_8gaussians(self):
        """Initialize 8 Gaussians dataset generator."""
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), 
                  (1./np.sqrt(2), 1./np.sqrt(2)), 
                  (1./np.sqrt(2), -1./np.sqrt(2)), 
                  (-1./np.sqrt(2), 1./np.sqrt(2)), 
                  (-1./np.sqrt(2), -1./np.sqrt(2))]
        self.centers = [(scale * x, scale * y) for x, y in centers]
        
        def sample_8gaussians(n):
            dataset = []
            for _ in range(n):
                point = self.rng.randn(2) * 0.5
                idx = self.rng.randint(8)
                center = self.centers[idx]
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            return np.array(dataset, dtype="float32") / 1.414
        
        self._sample_fn = sample_8gaussians
    
    def _init_pinwheel(self):
        """Initialize pinwheel dataset generator."""
        self.radial_std = 0.3
        self.tangential_std = 0.1
        self.num_classes = 5
        self.rate = 0.25
        self.rads = np.linspace(0, 2 * np.pi, self.num_classes, endpoint=False)
        
        def sample_pinwheel(n):
            num_per_class = n // self.num_classes
            features = self.rng.randn(self.num_classes*num_per_class, 2) * \
                np.array([self.radial_std, self.tangential_std])
            features[:, 0] += 1.
            labels = np.repeat(np.arange(self.num_classes), num_per_class)
            angles = self.rads[labels] + self.rate * np.exp(features[:, 0])
            rotations = np.stack([
                np.cos(angles), -np.sin(angles), 
                np.sin(angles), np.cos(angles)])
            rotations = np.reshape(rotations.T, (-1, 2, 2))
            return 2 * self.rng.permutation(np.einsum("ti,tij->tj", features, rotations))
        
        self._sample_fn = sample_pinwheel
    
    def _init_2spirals(self):
        """Initialize 2 spirals dataset generator."""
        def sample_2spirals(n):
            n_samples = n // 2
            n_vals = np.sqrt(self.rng.rand(n_samples, 1)) * 540 * (2 * np.pi) / 360
            d1x = -np.cos(n_vals) * n_vals + self.rng.rand(n_samples, 1) * 0.5
            d1y = np.sin(n_vals) * n_vals + self.rng.rand(n_samples, 1) * 0.5
            x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
            return x + self.rng.randn(*x.shape) * 0.1
        
        self._sample_fn = sample_2spirals
    
    def _init_checkerboard(self):
        """Initialize checkerboard dataset generator."""
        def sampling_checkerboard(n):
            x1 = self.rng.rand(n) * 4 - 2
            x2 = self.rng.rand(n) - self.rng.randint(0, 2, n) * 2 + (np.floor(x1) % 2)
            return np.concatenate([x1[:, None], x2[:, None]], 1) * 2
        self._sample_fn = sampling_checkerboard
        
    def _init_tree(self):
        sampler = TreeGaussianSampler(scale=np.array([4., 4.]), device="cpu")
        def sampling_tree(n):
            return sampler.sample(batch_size=n)
        self._sample_fn = sampling_tree
    
    def _init_real_dataset(self):
        """Initialize real dataset loader (BSDS300, POWER, etc.)."""
        dataset_map = {
            'bsds300': BSDS300,
            'power': POWER,
            'gas': GAS,
            'hepmass': HEPMASS,
            'miniboone': MINIBOONE
        }
        self.dataset = dataset_map[self.data_type]()
        self.dataset.trn.x = torch.from_numpy(self.dataset.trn.x)
        self.dataset.val.x = torch.from_numpy(self.dataset.val.x)
        self.dataset.tst.x = torch.from_numpy(self.dataset.tst.x)
        self._sample_fn = None  # Real datasets have predefined splits
    
    def sample(self, num_samples=1000):
        data = self._sample_fn(num_samples)
        if isinstance(data, torch.Tensor):
            return data.detach().float() 
        else:
            return torch.tensor(data).float() 
        
# In[] likelihood estimation: image
def add_noise(x):
    """[0, 1] -> [0, 255] -> add noise -> [0, 1]"""
    noise = torch.rand_like(x)
    x = x.mul(255).add_(noise).div_(256)
    return x

# In[] utils for MI estimation
def sample_joint_marginal_batch(data, batch_size): 
    dim = data.size(1) // 2
    N = data.size(0)
    device = data.device
    
    # joint samples
    joint_idx = torch.randperm(N, device=device)[:batch_size]
    joint_batch = data[joint_idx]
    
    # marginal samples
    idx_x = torch.randint(0, N, (batch_size,), device=device)
    idx_y = torch.randint(0, N, (batch_size,), device=device)
    X = data[idx_x][:, :dim]
    Y = data[idx_y][:, dim:]
    
    marginal_batch = torch.cat([X, Y], dim=1)
    
    return joint_batch, marginal_batch

def sample_joint_marginal(data): 
    dim = data.size(1) // 2
    N = data.size(0)
    device = data.device
    
    # joint samples
    joint_idx = torch.randperm(N, device=device)
    joint_data = data[joint_idx]
    
    # marginal samples
    idx_x = torch.randperm(N, device=device) 
    idx_y = torch.randperm(N, device=device) 
    X = data[idx_x][:, :dim]
    Y = data[idx_y][:, dim:]
    
    marginal_data = torch.cat([X, Y], dim=1)
    
    return joint_data, marginal_data

def sample_gaussian(rng, batch_size, mean, cov_matrix):
    samples = rng.multivariate_normal(mean=mean, cov=cov_matrix, size=batch_size)
    return torch.from_numpy(samples).type(torch.float32)

def generate_MI_data(rho, batch_size=1000, batch_num=4, d=2, data_type='gauss', rng=None):
    if rng is None:
        rng = np.random.RandomState()

    sample_num = batch_size * batch_num
    if data_type == 'gauss':
        # Gaussian distribution
        sample = torch.from_numpy(rng.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]], size=sample_num))
        joint_data, marginal_data = sample_joint_marginal(sample)
        mi = -np.log(1 - rho**2) * 0.5

    elif data_type == 'half_cube_map':
        # Half-cube mapping applied to Gaussian data
        sample = rng.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]], size=sample_num)
        x = sample[:, 0]
        y = sample[:, 1]
        x = np.sign(x) * np.abs(x)**(3/2)
        sample = np.stack((x, y), axis=1)
        sample = torch.from_numpy(sample)
        joint_data, marginal_data = sample_joint_marginal(sample)
        mi = -np.log(1 - rho**2) * 0.5

    elif data_type == 'asinh_mapping':
        # Asinh mapping applied to Gaussian data
        sample = rng.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]], size=sample_num)
        x = sample[:, 0]
        y = sample[:, 1]
        x = np.log(x + np.sqrt(1 + x**2))
        sample = np.stack((x, y), axis=1)
        sample = torch.from_numpy(sample)
        joint_data, marginal_data = sample_joint_marginal(sample)
        mi = -np.log(1 - rho**2) * 0.5

    elif data_type == 'additive_noise':
        # Additive noise model
        x = rng.uniform(0, 1, batch_size)
        noise = rng.uniform(-rho, rho, batch_size)
        y = x + noise
        sample = np.stack((x, y), axis=1)
        sample = torch.from_numpy(sample)
        joint_data, marginal_data = sample_joint_marginal(sample)
        if rho <= 0.5:
            mi = rho - np.log(2 * rho)
        else:
            mi = 1 / (4 * rho)

    elif data_type == 'gamma_exponential':
        # Gamma-exponential distribution
        x_samples = gamma.rvs(rho, size=sample_num, random_state=rng)
        y_samples = np.random.exponential(scale=1 / x_samples, size=sample_num)
        sample = np.stack((x_samples, y_samples), axis=1)
        sample = torch.from_numpy(sample)
        joint_data, marginal_data = sample_joint_marginal(sample)
        mi = digamma(rho + 1) - np.log(rho)

    elif data_type == "diag_multi_gauss":
        mi = 2 if d <= 10 else d // 4
        mi = float(mi)
        rho = (1 - np.exp(-(4 * mi) / d)) ** 0.5
        mean = np.zeros(d)
        cov_matrix = block_diag(*[[[1, rho], [rho, 1]] for _ in range(d // 2)])
        denom_cov_matrix = np.eye(d)
        joint_data = sample_gaussian(rng, batch_size, mean, cov_matrix)
        marginal_data = sample_gaussian(rng, batch_size, mean, denom_cov_matrix)

    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    return joint_data.type(torch.float32), marginal_data.type(torch.float32), torch.tensor(mi)

# In[] utils for image datasets in MI estimation
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, dataname="imdb.bert-imdb-finetuned", root="dataset", n_sample=None):
        super(TextDataset, self).__init__()

        root = os.path.join(root, dataname)

        self.classes = [0, 1]

        self.data = []
        self.counts = dict()
        self.label = []
        for idx, subclass in enumerate(self.classes):
            file_list = glob.glob(os.path.join(root, str(subclass), '*.npy'))
            file_list.sort()

            if (not n_sample in [None, "None"]):
                if n_sample > 0:
                    file_list = file_list[:n_sample]

            self.counts[idx] = len(self.data)
            self.data += [(filename, idx) for filename in file_list]
            self.label += [idx] * len(file_list)
        self.counts[len(self.classes)] = len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        filename, target_idx = self.data[index]
        data = np.load(filename)

        return data, target_idx
   
class MIRealDatasetGenerator:
    """Generator for real-world datasets with adjustable true MI values."""
    
    def __init__(self, data="image", dname1="mnist", dname2="imdb.bert-imdb-finetuned",
                 ds=10, dr=10, image_channels: int = 1, image_patches=[1, 2, 5], nuisance=0.0, data_dir="./data/"):
        
        self.data = data
        self.dname1 = dname1
        self.dname2 = dname2
        self.ds = ds
        self.dr = dr
        self.image_channels = image_channels
        self.image_patches = image_patches
        self.nuisance = nuisance
        self.data_dir = data_dir
        
        self.to_image = transforms.ToPILImage()
        self._init_datasets()

    def _init_datasets(self):
        """Initialize the required datasets."""
        if self.data in ["image", "mixture"]:
            self.images, self.idx_dict = self.image_subset(self.dname1, subclass_list=[0, 1], grayscale=False)
            self.img_size = int(np.sqrt(self.dr // self.image_channels))
        
        if self.data in ["text", "mixture"]:
            self.texts = TextDataset(dataname=self.dname2)
            
        if self.nuisance > 0:
            self.background, _ = self.image_subset("cifar10", np.arange(10), grayscale=False)
    
    def cal_bsc(self, ds, target_mi):
        """Calculate BSC probability for target MI value."""
        if target_mi >= ds:
            return 0.0
        
        # Binary entropy function
        def h_binary(p):
            return -p * np.log2(p) - (1-p) * np.log2(1-p) if 0 < p < 1 else 0
        
        # Solve for beta using binary search
        low, high = 0.0, 0.5
        tolerance = 1e-6
        max_iter = 100
        
        for _ in range(max_iter):
            beta = (low + high) / 2
            current_mi = ds * (1 - h_binary(beta))
            
            if abs(current_mi - target_mi) < tolerance:
                return beta
            elif current_mi < target_mi:
                high = beta
            else:
                low = beta
        
        return (low + high) / 2
        
    def apply_background(self, img_size, batch_size, x1, x2, eta, output_channels=1):
        """Apply background noise to image batches."""
        bg_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
        bg_idx = np.random.choice(len(self.background), 2 * batch_size)
        bg_batch = []
        for i in bg_idx:
            bg_batch.append(bg_transform(self.background.data[i])[None, :, :, :])
        bg_batch = torch.cat(bg_batch)
        z1 = torch.clip(x1 + bg_batch[:batch_size] * eta, 0, 1)
        z2 = torch.clip(x2 + bg_batch[batch_size:] * eta, 0, 1)
        
        if output_channels == 1:
            return (z1.mean(1, keepdims=True), z2.mean(1, keepdims=True))
        else:
            return (z1, z2)
        
    def image_subset(self, data_type, subclass_list, grayscale=False):
        """Load and preprocess image dataset subset."""
        if data_type == "mnist":
            transform = [transforms.Resize(size=(28, 28), interpolation=transforms.InterpolationMode.BICUBIC)]
            if grayscale:
                transform += [transforms.Grayscale()]
            transform += [transforms.ToTensor()]
            mnist = dset.MNIST(root=self.data_dir, train=True, download=True,
                              transform=transforms.Compose(transform))
            len_data = len(mnist)
            idx_per_digit = {}
            for digit in subclass_list:
                idx = mnist.targets == digit
                idx_per_digit.update({digit: torch.where(idx)[0].numpy()})
            return mnist, idx_per_digit
        
        elif data_type == "cifar10":
            transform = [transforms.Resize(size=(32, 32), interpolation=transforms.InterpolationMode.BICUBIC)]
            if grayscale:
                transform += [transforms.Grayscale()]
            transform += [transforms.ToTensor()]
            cifar = dset.CIFAR10(root=self.data_dir, train=True, download=True,
                                 transform=transforms.Compose(transform))
            idx_per_image = {}
            for c in subclass_list:
                idx = [t == c for t in cifar.targets]
                idx_per_image.update({c: np.where(idx)[0]})
            return cifar, idx_per_image
        
        elif data_type == "cifar100":
            transform = [transforms.Resize(size=(32, 32), interpolation=transforms.InterpolationMode.BICUBIC)]
            if grayscale:
                transform += [transforms.Grayscale()]
            transform += [transforms.ToTensor()]
            cifar = dset.CIFAR100(root=self.data_dir, train=True, download=True,
                                  transform=transforms.Compose(transform))
            idx_per_image = {}
            for c in subclass_list:
                idx = [t == c for t in cifar.targets]
                idx_per_image.update({c: np.where(idx)[0]})
            return cifar, idx_per_image
        
        else:
            raise ValueError(f"Unsupported dataset: {data_type}")
        
    def get_image(self, img_size, images, idx_dict, n_patch, bsc_p=0, return_label=False):
        """Generate image pairs with optional BSC noise.
        
        Args:
            img_size: Target output image size (img_size, img_size)
            images: Dictionary mapping image indices to [C, H, W] tensors
            idx_dict: Dictionary mapping class labels to lists of image indices
            n_patch: Patch grid dimensions [d1, d2, d3] for 3D patch arrangement
            bsc_p: Binary symmetric channel noise probability
            return_label: Whether to return patch labels
        
        Returns:
            tuple: (image1, image2) or (image1, image2, idx, idx_2)
        """
        # Convert n_patch to appropriate shape if it's a list
        if isinstance(n_patch, list):
            n_patch = torch.Size(n_patch)
        
        # Generate random binary indices for patch selection
        # idx: [d1, d2, d3] binary tensor for first image patch classes
        idx = torch.ones(size=n_patch) * 0.5
        idx = torch.bernoulli(idx)
        
        # Generate BSC noise mask
        # bsc_idx: [d1, d2, d3] noise mask for introducing differences
        bsc_idx = torch.ones(size=n_patch) * bsc_p
        bsc_idx = torch.bernoulli(bsc_idx)
        # Apply BSC noise to create second image's patch classes
        # idx_2: [d1, d2, d3] binary tensor for second image patch classes
        idx_2 = torch.abs(idx - bsc_idx)
        # Initialize lists to store image patches, im1, im2 will eventually contain [C, d2*H*d1, d3*W*d1] tensors
        im1, im2 = [], []
        # First dimension loop (d1) - controls depth/stacking
        for p1 in range(n_patch[0]):
            im1_1, im2_1 = [], []
            # Second dimension loop (d2) - controls vertical arrangement
            for p2 in range(n_patch[1]):
                im1_2, im2_2 = [], []
                # Third dimension loop (d3) - controls horizontal arrangement
                for p3 in range(n_patch[2]):
                    # Get class indices for current patch position
                    i = int(idx[p1, p2, p3])      # Class for first image patch
                    j = int(idx_2[p1, p2, p3])    # Class for second image patch
                    
                    # Get the actual class keys from idx_dict
                    class_keys = list(idx_dict.keys())
                    i_class = class_keys[i]       # Actual class label for first image
                    j_class = class_keys[j]       # Actual class label for second image
                    
                    # Randomly select images from respective classes
                    img1 = np.random.choice(idx_dict[i_class])
                    img2 = np.random.choice(idx_dict[j_class])
                    
                    # Ensure different images are selected
                    while img2 == img1:
                        img2 = np.random.choice(idx_dict[j_class])
                    # Append individual patches [C, H, W]
                    im1_2.append(images[img1][0])   # images[img1][0]: [C, H, W]
                    im2_2.append(images[img2][0])   # images[img2][0]: [C, H, W]
                
                # Horizontal concatenation along width dimension (dim=-1), Result: [C, H, d3*W] tensor for each row
                im1_1.append(torch.cat(im1_2, dim=-1))
                im2_1.append(torch.cat(im2_2, dim=-1))
            # Vertical concatenation along height dimension (dim=1), Result: [C, d2*H, d3*W] tensor for each depth slice
            im1.append(torch.cat(im1_1, dim=1))
            im2.append(torch.cat(im2_1, dim=1))
        # Depth concatenation along channel dimension (dim=0), Final result: [C*d1, d2*H, d3*W] tensor
        # Convert tensors to PIL images, Input: [C*d1, d2*H, d3*W] tensor, Output: PIL Image with size (C*d1, d3*W, d2*H)
        im1_pil = self.to_image(torch.cat(im1))
        im2_pil = self.to_image(torch.cat(im2))
        
        # Resize from (C*d1, d3*W, d2*H) to (C*d1, img_size, img_size), Final output: [C*d1, img_size, img_size] tensors
        resize = transforms.Compose([
            transforms.Resize(size=(img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])

        if return_label:
            return resize(im1_pil), resize(im2_pil), idx, idx_2
        else:
            return resize(im1_pil), resize(im2_pil)
    
    def get_text(self, dataset, idx_dict, n_text, bsc_p=0, get_label=False):
        """Generate text pairs with optional BSC noise."""
        idx = torch.ones(size=(n_text,)) * 0.5
        idx = torch.bernoulli(idx)
        bsc_idx = torch.ones(size=(n_text,)) * bsc_p
        bsc_idx = torch.bernoulli(bsc_idx)
        idx_2 = torch.abs(idx - bsc_idx)
        
        x1, x2 = [], []
        for t in range(n_text):
            i = int(idx[t])
            j = int(idx_2[t])
            
            text1 = np.random.choice(np.arange(idx_dict[i], idx_dict[i+1]))
            text2 = np.random.choice(np.arange(idx_dict[j], idx_dict[j+1]))
            
            while text2 == text1:
                text2 = np.random.choice(np.arange(idx_dict[j], idx_dict[j+1]))
                
            x1.append(torch.Tensor(dataset[text1][0]))
            x2.append(torch.Tensor(dataset[text2][0]))
            
        if get_label:
            return torch.cat(x1, dim=-1), torch.cat(x2, dim=-1), idx, idx_2
        else:
            return torch.cat(x1, dim=-1), torch.cat(x2, dim=-1)
        
    def image_batch(self, img_size, images, idx_dict, n_patch, bsc_p=0, batch_size=64, return_label=False):
        """Generate a batch of image pairs."""
        image1, image2 = [], []
        if return_label:
            label1, label2 = [], []
            
        for b in range(batch_size):
            if return_label:
                x1, x2, y1, y2 = self.get_image(img_size, images, idx_dict, n_patch, bsc_p=bsc_p, return_label=True)
                y1 = y1.numpy().reshape([-1]).tolist()
                y1 = [int(y) for y in y1]
                y2 = y2.numpy().reshape([-1]).tolist()
                y2 = [int(y) for y in y2]
                label1.append(int("".join(map(str, y1)), 2))
                label2.append(int("".join(map(str, y2)), 2))
            else:
                x1, x2 = self.get_image(img_size, images, idx_dict, n_patch, bsc_p=bsc_p)
                
            image1.append(x1[None, :, :, :])
            image2.append(x2[None, :, :, :])
            
        if return_label:
            return torch.cat(image1), torch.cat(image2), label1, label2
        else:
            return torch.cat(image1), torch.cat(image2)
        
    def text_batch(self, dataset, n_text, bsc_p=0, batch_size=64, n_sample=None, get_label=False):
        """Generate a batch of text pairs."""
        if n_sample is None:
            idx_dict = dataset.counts
        else:
            idx_dict = {0: 0, 1: n_sample, 2: dataset.counts[1] + n_sample}
            
        image1, image2 = [], []
        label1, label2 = [], []
        for b in range(batch_size):
            if get_label:
                x1, x2, y1, y2 = self.get_text(dataset, idx_dict, n_text, bsc_p=bsc_p, get_label=True)
                
                y1 = y1.numpy().reshape([-1]).tolist()
                y1 = [int(y) for y in y1]
                
                y2 = y2.numpy().reshape([-1]).tolist()
                y2 = [int(y) for y in y2]
                
                label1.append(int("".join(map(str, y1)), 2))
                label2.append(int("".join(map(str, y2)), 2))
            else:
                x1, x2 = self.get_text(dataset, idx_dict, n_text, bsc_p=bsc_p, get_label=False)
            image1.append(torch.Tensor(x1).view(1, -1))
            image2.append(torch.Tensor(x2).view(1, -1))
        
        if get_label:
            return torch.cat(image1), torch.cat(image2), label1, label2
        else:
            return torch.cat(image1), torch.cat(image2)
    
    def generate_batch(self, batch_size: int = 64, target_mi: Optional[float] = None,
                      bsc_p: Optional[float] = None, return_labels: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of data pairs (X, Y)."""
        
        if target_mi is not None:
            bsc_p = self.cal_bsc(self.ds, target_mi)
        
        if self.data == "image":
            if return_labels:
                x_batch, y_batch, label1, label2 = self.image_batch(
                    self.img_size, self.images, self.idx_dict, self.image_patches, 
                    bsc_p=bsc_p, batch_size=batch_size, return_label=True)
            else:
                x_batch, y_batch = self.image_batch(
                    self.img_size, self.images, self.idx_dict, self.image_patches, 
                    bsc_p=bsc_p, batch_size=batch_size)
                
        elif self.data == "text":
            if return_labels:
                x_batch, y_batch, label1, label2 = self.text_batch(
                    self.texts, self.ds, bsc_p=bsc_p, batch_size=batch_size, get_label=True)
            else:
                x_batch, y_batch = self.text_batch(
                    self.texts, self.ds, bsc_p=bsc_p, batch_size=batch_size)
        
        # Apply nuisance if specified
        if self.nuisance > 0:
            x_batch, y_batch = self.apply_background(
                self.img_size, batch_size, x_batch, y_batch, self.nuisance, 
                output_channels=self.image_channels)
        
        if return_labels:
            return x_batch, y_batch, label1, label2
        return x_batch, y_batch


# In[] utils for hand-crafted data in f-div
def true_kl_divergence(mean_0, cov_0, mean_1, cov_1):
    # cal KL(P||Q), P=N(mean_0, cov_0), Q=N(mean_1, cov_1)
    inv_covt = np.linalg.inv(cov_1)
    diff_mean = mean_1 - mean_0
    kl = 0.5 * (np.log(np.linalg.det(cov_1) / np.linalg.det(cov_0)) -
                len(mean_0) +
                np.trace(inv_covt @ cov_0) +
                diff_mean.T @ inv_covt @ diff_mean)
    return kl

def beta_kl_divergence(alpha_0, beta_0, alpha_t, beta_t):
    kl = betaln(alpha_t, beta_t) - betaln(alpha_0, beta_0)
    kl += (alpha_0 - alpha_t) * psi(alpha_0) + (beta_0 - beta_t) * psi(beta_0)
    kl += (alpha_t - alpha_0 + beta_t - beta_0) * psi(alpha_0 + beta_0)
    return kl

class DistributionShiftDatasetGenerator:
    """
    A professional dataset generator for distribution shift experiments,
    using strategy pattern with internal methods to avoid if-else branches.

    Supported data_type:
        - 'gauss_shift': linear mean/cov shift (your original)
        - 'beta_shift': beta distribution parameter drift
        - 'cifar_c': CIFAR-10-C style Gaussian noise (ICLR 2019): Hendrycks & Dietterich (2019), “Benchmarking Neural Network Robustness to Common Corruptions and Perturbations
        - 'domainbed': DomainBed covariance scaling (ICLR 2021): Gulrajani & Lopez-Paz (2020), In Search of Lost Domain Generalization
        - 'gauss_shift_kl': fixed +1.0 KL per step

    Automatic Shift Control:
        - Each call to .sample() increments an internal counter.
        - When (counter + 1) % reload_interval == 0, step is incremented by 1.
        - This allows smooth, controlled shifts during training (e.g., every 100 batches).

    Usage:
        gen = DistributionShiftDatasetGenerator(
            data_type='cifar_c',
            d=10,
            args={'max_severity': 0.4, 'reload_interval': 100}
        )
        for i in range(1000):
            xt, x0, kl = gen.sample(batch_size=64) 
    """

    _supported_modes = {'gauss_shift', 'beta_shift', 'cifar_c', 'domainbed', 'gauss_shift_kl'}

    def __init__(self, data_type='gauss_shift', d=2, rng=None, args=None):

        if data_type not in self._supported_modes:
            raise ValueError(f"Unsupported data_type: {data_type}. "
                             f"Choose from: {self._supported_modes}")

        self.data_type = data_type
        self.d = d
        self.rng = rng if rng is not None else np.random.RandomState()
        self.args = args
        
        def _get_arg(k, default):
            if args is None:
                return default
            if isinstance(args, dict):
                return args.get(k, default)
            return getattr(args, k, default)

        defaults = {
            'gauss_shift': {'delta_mu': 0.2, 'delta_sigma': 0.05},
            'beta_shift': {'alpha0': 1.0, 'beta0': 1.0, 'delta_mu': 2, 'delta_sigma': 0.1},
            'cifar_c': {'max_severity': 15.0},
            'domainbed': {'max_lambda': 10.0},
            'gauss_shift_kl': {},
        }
        mode_defaults = defaults[self.data_type]
        self.params = {k: _get_arg(k, v) for k, v in mode_defaults.items()}
        self.reload_interval = _get_arg('reload_interval', 100)

        self._sample_count = 0
        sample_method_name = f"_sample_{self.data_type}"
        self._sample_fn = getattr(self, sample_method_name)
        
        self.set_step(0)
        
    def set_step(self, step: int):
        """Set generator's step (called by DataModule / Callback)."""
        self.step = int(step)
        self._update_true_kl()
    
    def get_current_step(self):
        return self.step

    def _sample_gauss_shift(self, batch_size):
        x0 = self.rng.multivariate_normal(self.params["mean_0"], self.params["cov_0"], batch_size)
        xt = self.rng.multivariate_normal(self.params["mean_t"], self.params["cov_t"], batch_size)
        return xt, x0
    
    def _sample_beta_shift(self, batch_size):
        x0 = beta.rvs(self.params['alpha0'], self.params['beta0'], size=(batch_size, self.d), random_state=self.rng)
        xt = beta.rvs(self.params["alpha_t"], self.params["beta_t"], size=(batch_size, self.d), random_state=self.rng)
        return xt, x0
    
    def _sample_cifar_c(self, batch_size):
        x0 = self.rng.multivariate_normal(np.zeros(self.d), np.eye(self.d), batch_size)
        xt = self.rng.multivariate_normal(np.zeros(self.d), self.params["sigma_sq"] * np.eye(self.d), batch_size)
        return xt, x0
    
    def _sample_domainbed(self, batch_size):
        x0 = self.rng.multivariate_normal(np.zeros(self.d), self.params["cov_0"], batch_size)
        xt = self.rng.multivariate_normal(np.zeros(self.d), self.params["cov_t"], batch_size)
        return xt, x0
    
    def _sample_gauss_shift_kl(self, batch_size):
        x0 = self.rng.multivariate_normal(np.zeros(self.d), np.eye(self.d), batch_size)
        xt = self.rng.multivariate_normal(self.params['mean_t'], np.eye(self.d), batch_size)
        return xt, x0
    
    def _update_true_kl(self):
        if self.data_type == 'gauss_shift':
            delta_mu = self.params['delta_mu']
            delta_sigma = self.params['delta_sigma']
            mean_0 = np.zeros(self.d)
            cov_0 = np.eye(self.d)
            mean_t = mean_0 + delta_mu * self.step
            cov_t = (1 - delta_sigma * self.step) * cov_0
            cov_t = np.maximum(cov_t, 1e-6 * np.eye(self.d))
            self.params["mean_t"] = mean_t
            self.params["cov_t"] = cov_t
            self.params["mean_0"] = mean_0
            self.params["cov_0"] = cov_0
            kl = true_kl_divergence(mean_t, cov_t, mean_0, cov_0)
            self.true_kl = torch.tensor(kl, dtype=torch.float32)

        elif self.data_type == 'beta_shift':
            alpha_0, beta_0 = self.params['alpha0'], self.params['beta0']
            alpha_t = max(alpha_0 + self.params['delta_mu'] * self.step, 0.1)
            beta_t = beta_0 + self.params['delta_sigma'] * self.step
            kl = beta_kl_divergence(alpha_t, beta_t, alpha_0, beta_0)
            self.params["alpha_t"] = alpha_t
            self.params["beta_t"] = beta_t
            self.true_kl = torch.tensor(kl, dtype=torch.float32)

        elif self.data_type == 'cifar_c':
            max_severity = self.params['max_severity']
            sigma_sq = 1.0 + (self.step / 50.0) * max_severity
            kl = 0.5 * self.d * (sigma_sq - 1 - np.log(sigma_sq + 1e-8))
            self.params["sigma_sq"] = sigma_sq
            self.true_kl = torch.tensor(kl, dtype=torch.float32)

        elif self.data_type == 'domainbed':
            max_lambda = self.params['max_lambda']
            lambda_t = (self.step / 5.0) * max_lambda
            sigma_sq_last = 1 + lambda_t
            kl = 0.5 * (np.log(sigma_sq_last) + lambda_t)
            cov_0 = np.eye(self.d)
            cov_t = cov_0.copy()
            cov_t[-1, -1] = sigma_sq_last
            self.params["cov_0"] = cov_0
            self.params["cov_t"] = cov_t
            
            self.true_kl = torch.tensor(kl, dtype=torch.float32)

        elif self.data_type == 'gauss_shift_kl':
            kl = float(self.step)
            mu_norm = np.sqrt(2 * kl)
            direction = self.rng.randn(self.d)
            direction /= np.linalg.norm(direction)
            mean_t = direction * mu_norm
            self.params['mean_t'] = mean_t
            
            self.true_kl = torch.tensor(kl, dtype=torch.float32)
    
    def sample(self, batch_size):
        """Sample a batch, step is taken from internal state unless manually overridden."""
        xt, x0 = self._sample_fn(batch_size)

        return torch.tensor(xt, dtype=torch.float32), torch.tensor(x0, dtype=torch.float32), self.true_kl


class ChangePointDetectionDataGenerator:
    """
    Change point detection data generator.
    Generates (px, qx) pairs where:
        - px: samples from the true data distribution (e.g., text, creditcard, GMM)
        - qx: samples from a reference noise distribution (Gaussian or Simplex)
    """

    _supported_modes = {'cpd_mc', 'cpd_text', 'cpd_creditcard'}
    
    def __init__(self, data_type='cpd_text', d=6, rng=None, args=None):
        if data_type not in self._supported_modes:
            raise ValueError(f"Unsupported data_type: {data_type}. "
                             f"Choose from: {self._supported_modes}")

        self.seed = getattr(args, 'seed', 0)
        self.data_type = data_type
        self.d = d
        self.rng = rng if rng is not None else np.random.RandomState(self.seed)
        self.args = args
        self.use_simplex = getattr(args, 'use_simplex', False)  # Default: Gaussian
        
        if self.data_type == "cpd_mc":
            def sampling_fn(batch_size):
                px = self.sample_monte_carlo(num_samples=batch_size, d=self.d, experiment_id=0)
                return px
            
        elif self.data_type == "cpd_text":
            def sampling_fn(batch_size):
                px = self.load_and_process_text_data()
                # Sample batch_size rows (with replacement if needed)
                if px.size(0) < batch_size:
                    # indices = torch.randint(0, px.size(0), (batch_size,))
                    px = px
                else:
                    indices = torch.randperm(px.size(0))[:batch_size]
                    px = px[indices]
                return px
        
        elif self.data_type == "cpd_creditcard":
            def sampling_fn(batch_size):
                px = self.load_and_process_creditcard_data()
                if px.size(0) < batch_size:
                    # indices = torch.randint(0, px.size(0), (batch_size,))
                    px = px
                else:
                    indices = torch.randperm(px.size(0))[:batch_size]
                    px = px[indices]
                return px
        
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")
        self.sampling_fn = sampling_fn

    def sample(self, batch_size: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch of (px, qx) pairs.
        
        Args:
            batch_size: Number of samples to generate.
            
        Returns:
            px: (batch_size, d) samples from data distribution.
            qx: (batch_size, d) samples from reference noise (Gaussian or Simplex).
        """
        # Step 1: Generate px from data distribution
        px = self.sampling_fn(batch_size)
        batch_size =  px.size(0)

        # Step 2: Generate qx from reference noise
        noise_dim = px.size(1)
        if self.use_simplex:
            # Method 1: Dirichlet(1) → uniform on simplex
            # alpha = torch.ones(self.noise_dim)
            # qx = torch.distributions.Dirichlet(alpha).sample((n,))
            
            # Method 2 (more stable): softmax of Gaussian noise
            gaussian_noise = torch.randn(batch_size, noise_dim)
            qx = F.softmax(gaussian_noise, dim=-1)  # shape: (n, d), sum=1, >=0
            
            px = F.softmax(px, dim=-1)
        else:
            # Standard Gaussian reference
            qx = torch.randn(batch_size, noise_dim)

        return px, qx
        
    def sample_monte_carlo(self, num_samples: int = 1000, d: int = 6, experiment_id: int = 0) -> torch.Tensor:
        """generate monte carlo data"""
        # generate two gaussian distributions
        pdf_h0 = self._create_gmm(d, k=3, seed=self.seed + experiment_id)
        pdf_h1 = self._create_gmm(d, k=3, seed=self.seed + experiment_id + 1000)
        
        # evaluate change point
        nc = int(num_samples * 0.5)
        
        # generate data
        data_before = pdf_h0.sample((nc,))    # p_0
        data_after = pdf_h1.sample((num_samples - nc,))     # p_1
        data = torch.cat([data_before, data_after], dim=0)
        
        return data
    
    def load_and_process_text_data(
            self,
            file_path: str = "./data/density_ratio_estimation/change_point_detection/Language Detection.csv",
            languages: List[str] = ['French', 'Malayalam', 'Arabic'],
            vector_size: int = 20,
            save_path: Optional[str] = './data/density_ratio_estimation/change_point_detection/Language Detection.npy'
        ) -> torch.Tensor:
        """
        Load and process text data using character-level Word2Vec (as in the official baseline).
        If the precomputed .npy file exists at save_path, load it directly to avoid reprocessing.
        """
        # If the precomputed .npy file exists, load it directly
        if save_path and os.path.exists(save_path):
            Y = np.load(save_path)  # This is saved as Y.transpose() in baseline
            Y = Y.T  # Restore to shape (num_samples, vector_size)
            return torch.tensor(Y, dtype=torch.float32)
        
        data = pd.read_csv(file_path)
        X = data["Text"]
        data_list = []

        # Preprocess text: remove symbols/numbers and convert to lowercase
        for text in X:
            text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
            text = re.sub(r'[[]]', ' ', text)
            text = text.lower()
            data_list.append(text)

        # Train Word2Vec model (note: sentences are strings → gensim treats each character as a token)
        w2v_model = Word2Vec(
            sentences=data_list,
            vector_size=vector_size,
            window=5,
            min_count=1,
            workers=4
        )
        w2v_words = list(w2v_model.wv.index_to_key)

        # Compute average Word2Vec vector for each sentence (averaging over characters)
        sent_vectors = []
        for sent in data_list:  # `sent` is a string
            sent_vec = np.zeros(vector_size)
            cnt_words = 0
            for word in sent:  # Iterates over characters, NOT words
                if word in w2v_words:
                    vec = w2v_model.wv[word]
                    sent_vec += vec
                    cnt_words += 1
            if cnt_words != 0:
                sent_vec /= cnt_words
            sent_vectors.append(sent_vec)

        # Replace text column with vectors
        data['Text'] = sent_vectors

        # Copy and select only the three target languages
        df = data.copy()
        French = df.loc[df["Language"] == 'French'].reset_index()
        Malayalam = df.loc[df["Language"] == 'Malayalam'].reset_index()
        Arabic = df.loc[df["Language"] == 'Arabic'].reset_index()

        # Concatenate the three language subsets
        data = pd.concat([French, Malayalam, Arabic]).reset_index()
        Y = data["Text"].tolist()
        Y = np.array(Y)

        # Save transposed array (to match baseline: np.save(..., Y.transpose()))
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, Y.T)

        # Convert to PyTorch tensor and return
        text_data = torch.tensor(Y, dtype=torch.float32)
        return text_data
    
    def load_and_process_creditcard_data(
            self,
            file_path: str = "./data/density_ratio_estimation/change_point_detection/creditcard.txt",
            features: List[str] = ["V1", "V2", "V3", "V4"],
            genuine_samples: int = 2000,
            save_path: Optional[str] = './data/density_ratio_estimation/change_point_detection/creditcard.npy'
        ) -> torch.Tensor:
        """
        Load and process credit card fraud data as in the official baseline.
        If the saved .npy file exists, load it directly instead of reprocessing.
        """
        # If precomputed .npy file exists, load and return immediately
        if save_path and os.path.exists(save_path):
            Y_transposed = np.load(save_path)  # shape: (num_features, num_samples)
            Y = Y_transposed.T  # restore to (num_samples, num_features)
            return torch.tensor(Y, dtype=torch.float32)

        data = pd.read_csv(file_path, header=0)
        # Split into fraud and genuine transactions
        df = data.copy()
        fraud = df.loc[df["Class"] == 1].reset_index()
        genuine = df.loc[df["Class"] == 0].reset_index()

        # Take first 2000 genuine transactions and split into two halves
        genuine_1 = genuine[0:1000]
        genuine_2 = genuine[1000:2000]

        # Insert fraud transactions between the two genuine blocks
        fraud_in_genuine = pd.concat([genuine_1, fraud, genuine_2]).reset_index()

        # Select only the specified feature columns
        fraud_in_genuine = fraud_in_genuine[features]
        Y = fraud_in_genuine.values  # Shape: (num_samples, num_features)

        # Convert to PyTorch tensor
        creditcard_data = torch.tensor(Y, dtype=torch.float32)

        # Save transposed array to match baseline: np.save(..., Y.transpose())
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, Y.T)

        return creditcard_data
    
    def _create_gmm(self, d: int, k: int, sigma: float = 1.0, seed: int = None):
        """Gaussian Mixture Model"""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = self.rng
            
        means = torch.tensor(rng.randn(k, d) * 2.0)
        covs = torch.stack([torch.eye(d) * sigma for _ in range(k)])
        
        weights = torch.softmax(torch.randn(k), dim=0)
        
        mix = torch.distributions.Categorical(weights)
        comp = torch.distributions.MultivariateNormal(means, covariance_matrix=covs)
        return torch.distributions.MixtureSameFamily(mix, comp)
