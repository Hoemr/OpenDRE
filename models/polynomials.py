# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math

# In[]
def compute_binomial(k, l):
    """
    Compute the binomial coefficient C(k, l) using the log-gamma function.

    Args:
        k (tensor): Total number of items.
        l (tensor): Number of items to choose.

    Returns:
        torch.Tensor: The binomial coefficient C(k, l).
    """
    # Use log-gamma to compute factorials and avoid overflow
    log_binomial = (
        torch.lgamma(k + 1.0) -
        (torch.lgamma(l + 1.0) + torch.lgamma(k - l + 1.0))
    )
    # Exponentiate to get the binomial coefficient
    binomial = torch.exp(log_binomial)
    return binomial

# In[] noise polynomials
def get_noise_function(noise_type):
    if noise_type == "sin":
        noise_function = lambda k, t : torch.sin(k * t) / k.pow(1.5)
        integral_function = lambda k : (1 - torch.cos(k)) / k.pow(2.5)
        grad_function_k = lambda k, t : torch.cos(k * t) / k.pow(0.5)
    elif noise_type == "tk":
        noise_function = lambda k, t : t ** k / k.pow(1.5)
        integral_function = lambda k : 1. / ((k+1) * k.pow(1.5))
        grad_function_k = lambda k, t : k * (t**(k-1)) / k.pow(1.5)
    elif noise_type == "zero":
        noise_function = lambda k, t : torch.zeros((t.size(0), k.shape[-1]), device=t.device)
        integral_function = lambda k : torch.zeros((1, k.shape[-1]), device=k.device)
        grad_function_k = lambda k, t : torch.zeros((t.size(0), k.shape[-1]), device=t.device)
    return noise_function, integral_function, grad_function_k

class NoisePolynomials(nn.Module):
    def __init__(self, K, noise_type="tk"):
        super().__init__()
        self.K = K
        self.register_buffer("k", torch.arange(1, K, dtype=torch.int).unsqueeze(0))   # shape : [1, K-1]
        self.noise_function, self.integral_function, self.grad_function_k = get_noise_function(noise_type)

        self.register_buffer("zero_tensor", torch.zeros((1,1), dtype=torch.float32))
        
        self.register_buffer("integrals", self._compute_integrals(K))
        # self.integrals = self._compute_integrals(K)
        
    def forward(self, t):
        """
        Compute the values of the noise functions at time t for the first K values of k.
        """
        values = self.noise_function(self.k, t)  # Shape (..., K-1) 
        zero_tensor = torch.full(t.shape, 0.).to(t)
        values = torch.cat((zero_tensor, values), dim=-1)
        return values.view(-1, self.K)
    
    def grad_function(self, t):
        values = self.grad_function_k(self.k, t)
        zero_tensor = torch.full(t.shape, 0.).to(t)
        values = torch.cat((zero_tensor, values), dim=-1)
        return values.view(-1, self.K)

    def _compute_integrals(self, K):
        integrals = self.integral_function(self.k)
        return torch.cat((self.zero_tensor, integrals), dim=-1)
    
# In[] base polynomials and orthogonal polynomials
class PolynomialBasis(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.K = K
        self.register_buffer("k", torch.arange(K, dtype=torch.int).unsqueeze(0))
        # self.k = torch.arange(K, dtype=torch.float32)
        self.integrals = self._compute_integrals()

    def forward(self, t):
        # t -> [t^0, t^1, ..., t^(K-1)]
        t_powers = t.unsqueeze(-1) ** self.k       # shape (..., K)
        return t_powers.view(-1, self.K)

    def _compute_integrals(self):
        integrals = 1.0 / (self.k + 1)
        return integrals
    
    def grad_function(self, t):
        return self.k * t.pow(self.k - 1)

class LegendrePolynomials(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.K = K
        self.register_buffer("k", torch.arange(K, dtype=torch.int).unsqueeze(0))
        self.register_buffer("coefficients", self._precompute_coefficients(K))
        self.integrals = self._compute_integrals(K)

    def forward(self, t):
        # Expand t to powers [t^0, t^1, ..., t^(K-1)]
        t_powers = t ** self.k  # Shape (batch_size, K)

        # Apply coefficients to compute Legendre values
        values = torch.matmul(t_powers, self.coefficients.T)  # Shape (..., K)

        return values.view(-1, self.K)
    
    def grad_function(self, t):
        """
        Compute the derivative of the shifted Legendre polynomials at t.
        Args:
            t: Tensor of shape [batch_size, 1], where t ∈ [0, 1].
        Returns:
            Tensor of shape [batch_size, K], containing the derivative values.
        """
        # Compute the Legendre polynomial values
        P_t = self.forward(t)  # Shape (batch_size, K)

        # Compute the derivatives using the recurrence relation
        grad_values = torch.zeros_like(P_t)  # Shape (batch_size, K)
        for k in range(1, self.K):  # P_0'(t) = 0, so start from k=1
            grad_values[:, k] = (2 * k - 1) * P_t[:, k - 1] + grad_values[:, k - 1]

        return grad_values
    
    def _compute_integrals(self, K):
        integrals = torch.zeros(K)
        integrals[0] = 1.0  # Integral of P_0(t) is 1
        # For k >= 1, the integral is 0
        return integrals

    def _compute_coefficient(self, k, l):
        # compute bio{k, l} * bio{k+l, l} * (-1)^{k-l}
        binomial_k_l = compute_binomial(k, l)
        binomial_k_plus_l_l = compute_binomial(k + l, l)
        return (-1) ** (k - l) * binomial_k_l * binomial_k_plus_l_l

    def _precompute_coefficients(self, K):
        coefficients = torch.zeros(K, K)  # Each row corresponds to a polynomial

        for k in self.k.squeeze():
            l = torch.arange(k + 1, dtype=torch.float32)
            coefficients[k, :k+1] = torch.tensor([self._compute_coefficient(k, l_val) for l_val in l])

        return coefficients
    
class LaguerrePolynomials(nn.Module):
    def __init__(self, K, alpha=0):
        super().__init__()
        self.K = K
        self.alpha = alpha
        self.register_buffer("k", torch.arange(K, dtype=torch.int).unsqueeze(0))  # Laguerre degree
        self.register_buffer("coefficients", self._precompute_coefficients(K))  # Precompute coefficients
        self.register_buffer("grad_coefficients", self._precompute_grad_coefficients(K))
        self.integrals = self._compute_integrals(K)

    def forward(self, t):
        # shape of t : [batch_size, 1]
        # Ensure t is in [0, 1] and compute the powers of log(t)
        t_powers = torch.log(t) ** self.k  # Shape (batch_size, K)
        
        # Compute the polynomial values by multiplying powers of log(t) with coefficients
        values = torch.matmul(t_powers, self.coefficients.T)  # Shape (..., K)
        
        return values.view(-1, self.K)  # Return the computed values for the Laguerre polynomials
    
    def grad_function(self, t):
        """
        Compute the derivative of Laguerre polynomials with respect to t.
        """
        # Compute the powers of log(t): (ln(t))^l
        log_t = torch.log(t)  # Shape [batch_size, 1]
        log_t_powers = log_t ** self.k  # Shape [batch_size, K]
    
        # Compute the derivative values
        grad_values = torch.matmul(log_t_powers, self.grad_coefficients.T) / t  # Shape [batch_size, K]

        return grad_values.view(-1, self.K)

    def _precompute_grad_coefficients(self, K):
        """
        Precompute the derivative coefficients for all k and l.
        The coefficients are given by:
            binomial(k + alpha, k - l) * (-1)^l * (k - l) / l!
        """
        grad_coefficients = torch.zeros(K, K)  # Shape (K, K)
        for k in self.k.squeeze():
            ls = torch.arange(k + 1, dtype=torch.int)
            for l in ls:  # Sum from l=0 to l=k-1
                binomial_k_plus_alpha_k_minus_l_minus_1 = compute_binomial(k + self.alpha, k - l - 1)
                factorial_l = torch.exp(torch.lgamma(l + 1))
                grad_coefficients[k, l] = binomial_k_plus_alpha_k_minus_l_minus_1 / factorial_l

        return grad_coefficients
    
    def _compute_integrals(self, K):    
        # Precompute binomial coefficients
        binomial_coeffs = torch.zeros(K, K + 1)  # Shape (K, K + 1)
        for k in self.k.squeeze():
            ls = torch.arange(k + 1, dtype=torch.int)
            for l in ls:
                binomial_coeffs[k, l] = compute_binomial(k + self.alpha, k - l)  # Use scipy's binom function
        
        # Compute integrals using vectorized operations
        l_indices = torch.arange(K + 1, dtype=torch.float32)  # l values for each k
        integrals = torch.sum(
            binomial_coeffs * ((-1) ** l_indices) / (l_indices + 1),  # Vectorized computation
            dim=1
        )
        
        return integrals

    def _compute_coefficient(self, k, l):
        # Binomial coefficient binomial(k + alpha, k - l) * (1/l!)
        binomial_k_plus_alpha_k_minus_l = compute_binomial(k + self.alpha, k - l)
        # binomial_k_l = self._compute_binomial(k, l)
        factorial_l = torch.exp(torch.lgamma(l + 1))
        return binomial_k_plus_alpha_k_minus_l * (1 / factorial_l)
    
    def _precompute_coefficients(self, K):
        coefficients = torch.zeros(K, K)  # Each row corresponds to a polynomial

        for k in self.k.squeeze():
            l = torch.arange(k + 1, dtype=torch.int)
            coefficients[k, :k+1] = torch.tensor([self._compute_coefficient(k, l_val) for l_val in l])

        return coefficients

class ChebyshevPolynomials(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.K = K  # Maximum degree of the polynomial
        self.register_buffer("k", torch.arange(K, dtype=torch.int).unsqueeze(0))  # Degrees of the polynomials
        self.register_buffer("coefficients", self._precompute_coefficients(K))  # Precompute polynomial coefficients
        self.register_buffer("grad_coefficients", self._precompute_grad_coefficients(K))  # Precompute derivative coefficients
        self.integrals = self._compute_integrals(K)  # Precompute integrals

    def forward(self, s):
        """
        Evaluate the transformed Chebyshev polynomials at s.
        s: Tensor of shape [batch_size, 1], where s ∈ [0, 1].
        """
        # Transform s ∈ [0, 1] to t ∈ [-1, 1]
        t = 2 * s - 1  # Shape [batch_size, 1]

        # Compute the Chebyshev polynomial values using the explicit form
        values = torch.cos(self.k * torch.acos(t))  # Shape [batch_size, K]

        return values.view(-1, self.K)

    def grad_function(self, s):
        """
        Compute the derivative of the transformed Chebyshev polynomials with respect to s.
        s: Tensor of shape [batch_size, 1], where s ∈ [0, 1].
        """
        # Transform s ∈ [0, 1] to t ∈ [-1, 1]
        t = 2 * s - 1  # Shape [batch_size, 1]

        # Compute the derivative using the second-kind Chebyshev polynomials
        epsilon = 1e-8
        grad_values = 2 * self.k * torch.sin(self.k * torch.acos(t)) / torch.sqrt(1 - t**2 + epsilon)  # Shape [batch_size, K]

        return grad_values.view(-1, self.K)

    def _precompute_grad_coefficients(self, K):
        """
        Precompute the coefficients for the derivative of the transformed Chebyshev polynomials.
        This is not directly used in the derivative computation but is included for consistency.
        """
        grad_coefficients = torch.zeros(K, K)  # Placeholder, as derivatives are computed directly
        return grad_coefficients

    def _compute_integrals(self, K):
        """
        Compute the integrals of the transformed Chebyshev polynomials over [0, 1].
        """
        integrals = torch.zeros(K)
        integrals[0] = 1.0  # Integral of P_0^*(s) = 1 over [0, 1] is 1
        # For k ≥ 1, the integral is 0 due to orthogonality
        return integrals

    def _precompute_coefficients(self, K):
        """
        Precompute the coefficients for the transformed Chebyshev polynomials.
        This is not directly used in the polynomial evaluation but is included for consistency.
        """
        coefficients = torch.zeros(K, K)  # Placeholder, as polynomials are computed directly
        return coefficients
    
class JacobiPolynomials(nn.Module):
    def __init__(self, K, alpha=1e-3, beta=1e-3):
        super().__init__()
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.register_buffer("k", torch.arange(K, dtype=torch.int).unsqueeze(0))  # Jacobi degree
        self.coefficients = self._precompute_coefficients(K)
        self.grad_coefficients = self._precompute_grad_coefficients(K)
        self.integrals = self._compute_integrals(K)

    def forward(self, t):
        # t_shifted = 2 * t - 1  # Transform domain from [0, 1] to [-1, 1]
        values = torch.zeros(t.size(0), self.K, device=t.device)
        
        batch_size = t.size(0)
        
        # Expand t to shape [batch_size, K, K]
        t_expanded = t.view(batch_size, 1, 1)  # Shape [batch_size, 1, 1]
        l_values = torch.arange(self.K, device=t.device).view(1, K, 1)  # Shape [1, K, 1]
        k_values = torch.arange(self.K, device=t.device).view(1, 1, K)  # Shape [1, 1, K]

        # Compute (x-1)^l and x^{k-l}
        term1 = (t_expanded - 1).pow(l_values)  # Shape [batch_size, K, 1]
        term2 = t_expanded.pow(k_values - l_values)  # Shape [batch_size, 1, K]

        # Combine terms into a single tensor
        terms = term1 * term2  # Shape [batch_size, K, K]

        # Expand coefficients to match the shape of terms
        coefficients_expanded = self.coefficients.unsqueeze(0)  # Shape [1, K, K]

        # Use einsum to compute the polynomial values
        values = torch.einsum("bkl,bkl->bk", terms, coefficients_expanded)  # Shape [batch_size, K]
        return values

    def grad_function(self, t):
        t_shifted = 2 * t - 1
        grad_values = torch.zeros(t.size(0), self.K, device=t.device)
        for k in self.k.squeeze():
            if k > 0:
                grad_values[:, k] = (k + self.alpha + self.beta + 1) * self.forward(t)[:, k - 1]
        return grad_values

    def _precompute_coefficients(self, K):
        coefficients = torch.zeros(K, K, device=self.k.device)  # Shape (K, K)
        for k in self.k.squeeze():
            l = torch.arange(k + 1, dtype=torch.float32, device=self.k.device)
            coefficients[k, :k+1] = torch.stack([
                compute_binomial(k + self.alpha, l_val) * compute_binomial(k + self.beta, k - l_val) / 2**k
                for l_val in l
            ])
        return coefficients

    def _precompute_grad_coefficients(self, K):
        grad_coefficients = torch.zeros(K, K, device=self.k.device)
        for k in self.k.squeeze():
            if k > 0:
                grad_coefficients[k, :k] = (k + self.alpha + self.beta + 1) * self.coefficients[k - 1, :k]
        return grad_coefficients

    def _compute_integrals(self, K):
        integrals = torch.zeros(K, device=self.k.device)
        integrals[0] = 1  # Integral of P_0 is 1
        for k in range(1, K):
            integrals[k] = 0  # Integral of higher-order Jacobi polynomials is 0
        return integrals
  
class BernsteinPolynomials(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.K = K
        self.register_buffer("k", torch.arange(K, dtype=torch.int).unsqueeze(0))  # Degrees of the polynomials
        self.register_buffer("coefficients", self._precompute_coefficients(K))  # Precompute polynomial coefficients
        self.integrals = self._compute_integrals(K)  # Precompute integrals

    def forward(self, t):
        t = t.clamp(0, 1)  # Ensure t ∈ [0, 1]
        k = self.k  # Shape [1, K]
        # Compute Bernstein polynomial values
        values = self.coefficients * t**k * (1 - t) ** (self.K - 1 - k)  # Shape [batch_size, K]
        return values

    def grad_function(self, t):
        t = t.clamp(0, 1)  # Ensure t ∈ [0, 1]
        k = self.k  # Shape [1, K]
        # Compute the derivative values
        grad_values = self.coefficients * (
            k * t ** (k - 1) * (1 - t) ** (self.K - 1 - k) -
            (self.K - 1 - k) * t**k * (1 - t) ** (self.K - 2 - k)
        )  # Shape [batch_size, K]
        return grad_values

    def _compute_integrals(self, K):
        integrals = torch.ones(K) / K  # All integrals are 1/K
        return integrals

    def _compute_coefficient(self, k, l):
        return compute_binomial(self.K - 1, k)  # Binomial coefficient C(K-1, k)

    def _precompute_coefficients(self, K):
        coefficients = torch.zeros(K)  # Each element corresponds to a polynomial
        for k in self.k.squeeze():
            coefficients[k] = compute_binomial(k, torch.tensor(0.))  # Bernstein coefficients are binomial coefficients
        return coefficients
    
class RBFPolynomials(nn.Module):
    """
    Base class for Radial Basis Function (RBF) polynomials.
    """

    def __init__(self, K, log_sigma=0.0, shape_type="uniform", alpha=1000.0, beta=1.0):
        super().__init__()
        self.K = K
        self.shape_type = shape_type
        self.alpha = alpha
        self.beta = beta

        # Define centers c_k and store as buffer
        self.register_buffer("c_k", torch.linspace(0, 1, K).unsqueeze(0))

        # Learnable parameter for log(sigma)
        # self.log_sigma = nn.Parameter(torch.full((1, K), log_sigma))
        self.log_sigma = nn.Parameter(torch.full((1, K), -math.log(K)))

        # Cache variables
        self.register_buffer("_cached_integral", None)
        self.cached_sigma = None
        self._cached_log_sigma_version = -1
        self.nu = torch.tensor(0.)
        
        if shape_type == "decay":
            self.register_buffer("K_tensor", torch.tensor(K+1, dtype=torch.float32))
            self.register_buffer("log_K", torch.log(self.K_tensor))
            self.register_buffer("k_values", torch.arange(1, K+1, dtype=torch.float32).view(1, K))
            self.register_buffer("sigmoid1", torch.sigmoid(self.alpha * (self.K_tensor - self.k_values)))
            self.register_buffer("sigmoid2", torch.sigmoid(self.alpha * (self.k_values - self.K_tensor)))   
            self._set_decay_params()   

    def _set_decay_params(self):
        raise NotImplementedError
    
    def _compute_sigma(self):
        raise NotImplementedError

    @property
    def sigma(self):
        if self.training:
            current_version = self.log_sigma._version
            if (self.cached_sigma is None) or (current_version != self._cached_log_sigma_version):
                self.cached_sigma = self._compute_sigma()
                self._cached_log_sigma_version = current_version
            return self.cached_sigma
        else:
            if self.cached_sigma is None:
                self.cached_sigma = self._compute_sigma()
            return self.cached_sigma

    def forward(self, t):
        raise NotImplementedError

    def grad_function(self, t):
        raise NotImplementedError

    def _compute_integrals(self):
        raise NotImplementedError

    @property
    def integrals(self):
        if self.training:
            self.reset_cache()
            return self._compute_integrals()
        else:
            if self._cached_integral is None:
                self._cached_integral = self._compute_integrals()
            return self._cached_integral

    def reset_cache(self):
        self._cached_integral = None
        
class GaussianRBFPolynomials(RBFPolynomials):
    """Gaussian Radial Basis Function (RBF) polynomials."""
    def __init__(self, K, log_sigma=-2., shape_type="uniform", alpha=1000.0, beta=1.0):
        super().__init__(K, log_sigma, shape_type, alpha, beta)
        self.register_buffer("sqrt_pi", torch.sqrt(torch.tensor(torch.pi)))
        
        if shape_type == "uniform":
            def sigma_fn():
                return torch.exp(self.log_sigma.clamp(min=-6.9)) + 1e-8
        elif self.shape_type == "decay":
            def sigma_fn():
                term1 = (self.beta * torch.exp(self.log_sigma - self.nu * self.log_K)) * self.sigmoid1
                term2 = self.beta * self.residual_term
                return term1 + term2
        else:  
            raise ValueError(f"Unknown shape_type: {self.shape_type}")
        self.sigma_fn = sigma_fn
        
    def _set_decay_params(self):
        self.gamma = torch.log(self.K_tensor)
        self.nu = nn.Parameter(torch.tensor(0.2).to(self.K_tensor))
        self.register_buffer("residual_term", torch.exp(-self.gamma * self.k_values) * self.sigmoid2)

    def _compute_sigma(self):
        return self.sigma_fn()

    def forward(self, t):
        sigma_squared = self.sigma ** 2
        squared_distances = (t - self.c_k) ** 2   # Shape [batch_size, K]
        return torch.exp(-squared_distances / sigma_squared)  # Shape [batch_size, K]

    def grad_function(self, t):
        sigma_squared = self.sigma ** 2
        squared_distances = (t - self.c_k) ** 2    # Shape [batch_size, K]
        exp_term = torch.exp(-squared_distances / sigma_squared)    # Shape [batch_size, K]
        return -2 * (t - self.c_k) / sigma_squared * exp_term    # Shape [batch_size, K]

    def _compute_integrals(self):
        sigma = self.sigma
        sigma_sqrt_pi = sigma * self.sqrt_pi
        erf_term_1 = torch.erf((1 - self.c_k) / sigma)
        erf_term_2 = torch.erf(self.c_k / sigma)
        return (sigma_sqrt_pi / 2) * (erf_term_1 + erf_term_2)
    
class IMQRBFPolynomials(RBFPolynomials):
    """Inverse Multiquadric (IMQ) Radial Basis Function polynomials."""
    def __init__(self, K, log_sigma=-2., shape_type="uniform", alpha=1000.0, beta=1.0):
        # log_sigma = 2.
        super().__init__(K, log_sigma, shape_type, alpha, beta)
        
        if shape_type == "uniform":
            def sigma_fn():
                return torch.exp(self.log_sigma.clamp(min=-6.9)) + 1e-8
        elif self.shape_type == "decay":
            def sigma_fn():
                term1 = (self.beta * torch.exp(self.log_sigma + self.nu * self.log_K)) * self.sigmoid1
                term2 = self.residual_term / self.beta
                return term1 + term2
        else:  
            raise ValueError(f"Unknown shape_type: {self.shape_type}")
        self.sigma_fn = sigma_fn

    def _set_decay_params(self):
        self.gamma = 5.0
        self.nu = nn.Parameter(torch.tensor(0.2))
        self.register_buffer("residual_term", (1. + self.k_values**2).pow(self.gamma) * self.sigmoid2)

    def _compute_sigma(self):
        return self.sigma_fn()

    def forward(self, t):
        sigma_squared = self.sigma ** 2
        squared_distances = (t - self.c_k) ** 2        # Shape [batch_size, K]
        denominator = squared_distances + sigma_squared
        return self.sigma / torch.sqrt(denominator.clamp(min=1e-8))        # Shape [batch_size, K]

    def grad_function(self, t):
        sigma_squared = self.sigma ** 2
        squared_distances = (t - self.c_k) ** 2          # Shape [batch_size, K]
        denominator = squared_distances + sigma_squared
        denominator = (denominator.clamp(min=1e-8)) ** (1.5)            # Shape [batch_size, K]
        return - self.sigma * (t - self.c_k) / denominator.clamp(min=1e-12)            # Shape [batch_size, K]

    def _compute_integrals(self):
        sigma_squared = self.sigma ** 2
        c_k = self.c_k
        term1 = (1 - c_k) + torch.sqrt((1 - c_k) ** 2 + sigma_squared)  # Shape [1, K]
        term2 = -c_k + torch.sqrt(c_k ** 2 + sigma_squared)  # Shape [1, K]
        return self.sigma * torch.log((term1 / term2.clamp(min=1e-8)).clamp(min=1e-8, max=1e8))
    
class MaternRBFPolynomials(RBFPolynomials):
    """Matérn (v=3/2) Radial Basis Function polynomials."""
    def __init__(self, K, log_sigma=-2., shape_type="uniform", alpha=1000.0, beta=1.0):
        super().__init__(K, log_sigma, shape_type, alpha, beta)
        self.register_buffer("sqrt3", torch.sqrt(torch.tensor(3.0)))
        
        if shape_type == "uniform":
            def sigma_fn():
                return torch.exp(self.log_sigma.clamp(min=-6.9))
        elif self.shape_type == "decay":
            raise NotImplementedError("Decay shape type not implemented for Matern RBF")
        else:  
            raise ValueError(f"Unknown shape_type: {self.shape_type}")
        self.sigma_fn = sigma_fn
        
    def _set_decay_params(self):
        pass

    def _compute_sigma(self):
        return self.sigma_fn()

    def forward(self, t):
        """
        Compute Matern RBF: g_k(t) = (1 + √3 * |t-c_k|/σ_k) * exp(-√3 * |t-c_k|/σ_k)
        """
        sigma = self.sigma
        abs_distances = torch.abs(t - self.c_k)  # Shape [batch_size, K]
        scaled_distances = self.sqrt3 * abs_distances / torch.clamp(sigma, min=1e-6) 
        exp_term = torch.exp(-scaled_distances.clamp(max=50))  
        return (1 + scaled_distances) * exp_term  # Shape [batch_size, K]

    def grad_function(self, t):
        """
        Compute derivative: g_k'(t) = -3(t-c_k)/σ_k² * exp(-√3 * |t-c_k|/σ_k)
        """
        sigma = self.sigma
        distances = t - self.c_k  # Shape [batch_size, K]
        abs_distances = torch.abs(distances)
        scaled_distances = self.sqrt3 * abs_distances / torch.clamp(sigma, min=1e-6)
        exp_term = torch.exp(-scaled_distances.clamp(max=50))
        denom = torch.clamp(sigma ** 2, min=1e-12)
        return -3 * distances / denom * exp_term  # Shape [batch_size, K]

    def _compute_integrals(self):
        """
        Compute integral: ∫_0^1 g_k(t) dt = 2σ_k/√3 - σ_k[( (1-c_k)/σ_k + 2/√3 )e^{-√3(1-c_k)/σ_k} 
                        + ( c_k/σ_k + 2/√3 )e^{-√3 c_k/σ_k} ]
        """
        sigma = self.sigma
        c_k = self.c_k
        sqrt3 = self.sqrt3
        
        # Precompute terms
        denom1 = torch.clamp(sigma, min=1e-6)
        term1_coeff = (1 - c_k) / denom1 + 2 / sqrt3
        exponent1 = (-sqrt3 * (1 - c_k) / denom1).clamp(max=50)
        term1_exp = torch.exp(exponent1)
        term1 = term1_coeff * term1_exp
        
        term2_coeff = c_k / denom1 + 2 / sqrt3
        exponent2 = (-sqrt3 * c_k / denom1).clamp(max=50)
        term2_exp = torch.exp(exponent2)
        term2 = term2_coeff * term2_exp
        
        return (2 * sigma / sqrt3) - sigma * (term1 + term2)

class WendlandRBFPolynomials(RBFPolynomials):
    """Wendland (C²) Radial Basis Function polynomials (compactly supported)."""
    def __init__(self, K, log_sigma=-2., shape_type="uniform", alpha=1000.0, beta=1.0):
        super().__init__(K, log_sigma, shape_type, alpha, beta)
        
        if shape_type == "uniform":
            def sigma_fn():
                return torch.exp(self.log_sigma) + 1e-8
        elif self.shape_type == "decay":
            raise NotImplementedError("Decay shape type not implemented for Wendland RBF")
        else:  
            raise ValueError(f"Unknown shape_type: {self.shape_type}")
        self.sigma_fn = sigma_fn
        
    def _set_decay_params(self):
        pass

    def _compute_sigma(self):
        return self.sigma_fn()

    def forward(self, t):
        """
        Compute Wendland RBF: g_k(t) = (1 - r)_+^4 * (4r + 1) where r = |t-c_k|/sigma_k
        """
        sigma = self.sigma
        abs_distances = torch.abs(t - self.c_k)  # Shape [batch_size, K]
        r = abs_distances / sigma  # Shape [batch_size, K]
        
        # Compact support: zero outside [c_k-σ_k, c_k+σ_k]
        mask = (r < 1.0).float()  # Shape [batch_size, K]
        one_minus_r = (1 - r) * mask
        return (one_minus_r ** 4) * (4 * r + 1) * mask  # Shape [batch_size, K]

    def grad_function(self, t):
        """
        Compute derivative: g_k'(t) = -20(t-c_k)/sigma_k² * (1 - r)_+^3 where r = |t-c_k|/sigma_k
        """
        sigma = self.sigma
        distances = t - self.c_k  # Shape [batch_size, K]
        abs_distances = torch.abs(distances)
        r = abs_distances / sigma  # Shape [batch_size, K]
        
        # Compact support: zero outside [c_k-σ_k, c_k+σ_k]
        mask = (r < 1.0).float()  # Shape [batch_size, K]
        one_minus_r = (1 - r) * mask
        
        return -20 * distances / (sigma ** 2) * (one_minus_r ** 3) * mask  # Shape [batch_size, K]

    def _compute_integrals(self):
        """
        Compute integral: ∫_0^1 g_k(t) dt by integrating over the effective support
        [max(0, c_k - sigma_k), min(1, c_k + sigma_k)]
        """
        sigma = self.sigma
        c_k = self.c_k
        
        # Determine integration bounds for each basis function
        lower_bounds = torch.clamp(c_k - sigma, min=0.0)  # Shape [1, K]
        upper_bounds = torch.clamp(c_k + sigma, max=1.0)  # Shape [1, K]
        
        # For Wendland RBF, we need to compute the integral numerically
        # since the closed form involves piecewise evaluation
        integrals = torch.zeros_like(c_k)
        
        # We'll use numerical integration with 100 points for accuracy
        num_points = 100
        for k in range(self.K):
            lower = lower_bounds[0, k].item()
            upper = upper_bounds[0, k].item()
            
            if lower >= upper:
                integrals[0, k] = 0.0
                continue
                
            # Create integration points
            t_points = torch.linspace(lower, upper, num_points, device=c_k.device)
            t_points = t_points.unsqueeze(1)  # Shape [num_points, 1]
            
            # Evaluate the basis function at these points
            abs_dist = torch.abs(t_points - c_k[0, k])
            r = abs_dist / sigma[0, k]
            mask = (r < 1.0).float()
            one_minus_r = (1 - r) * mask
            values = (one_minus_r ** 4) * (4 * r + 1) * mask
            
            # Numerical integration using trapezoidal rule
            integral = torch.trapezoid(values.flatten(), t_points.flatten())
            integrals[0, k] = integral
            
        return integrals

class RationalQuadraticRBFPolynomials(RBFPolynomials):
    """Rational Quadratic (RQ) Radial Basis Function polynomials."""
    def __init__(self, K, log_sigma=-2., shape_type="uniform", alpha=1000.0, beta=1.0):
        super().__init__(K, log_sigma, shape_type, alpha, beta)
        
        if shape_type == "uniform":
            def sigma_fn():
                return torch.exp(self.log_sigma.clamp(min=-6.9))
        elif self.shape_type == "decay":
            raise NotImplementedError("Decay shape type not implemented for RQ RBF")
        else:  
            raise ValueError(f"Unknown shape_type: {self.shape_type}")
        self.sigma_fn = sigma_fn
        
    def _set_decay_params(self):
        pass

    def _compute_sigma(self):
        return self.sigma_fn()

    def forward(self, t):
        """
        Compute RQ RBF: g_k(t) = σ_k² / ((t-c_k)² + σ_k²)
        """
        sigma = self.sigma
        sigma_squared = sigma ** 2
        squared_distances = (t - self.c_k) ** 2  # Shape [batch_size, K]
        denominator = squared_distances + sigma_squared
        denominator = torch.clamp(denominator, min=1e-12)
        return sigma_squared / denominator  # Shape [batch_size, K]

    def grad_function(self, t):
        """
        Compute derivative: g_k'(t) = -2σ_k²(t-c_k) / ((t-c_k)² + σ_k²)²
        """
        sigma = self.sigma
        sigma_squared = sigma ** 2
        distances = t - self.c_k  # Shape [batch_size, K]
        squared_distances = distances ** 2  # Shape [batch_size, K]
        inner_denom = squared_distances + sigma_squared
        inner_denom = torch.clamp(inner_denom, min=1e-8)
        denominator = inner_denom ** 2
        denominator = torch.clamp(denominator, min=1e-12)
        return -2 * sigma_squared * distances / denominator  # Shape [batch_size, K]

    def _compute_integrals(self):
        """
        Compute integral: ∫_0^1 g_k(t) dt = σ_k [arctan((1-c_k)/σ_k) + arctan(c_k/σ_k)]
        """
        sigma = self.sigma
        c_k = self.c_k
        safe_sigma = torch.clamp(sigma, min=1e-6)
        
        term1 = torch.atan((1 - c_k) / safe_sigma)
        term2 = torch.atan(c_k / safe_sigma)
        
        return sigma * (term1 + term2)

POLYNOMIALS = {
    "polynomials": PolynomialBasis,
    "laguerre": LaguerrePolynomials,
    "legendre": LegendrePolynomials,
    "chebyshev": ChebyshevPolynomials,
    "jacobi": JacobiPolynomials,
    "bernstein": BernsteinPolynomials,
    "gaussianrbf": GaussianRBFPolynomials,
    "imqrbf": IMQRBFPolynomials,
    "maternrbf": MaternRBFPolynomials,
    "wendlandrbf": WendlandRBFPolynomials,
    "rqrbf": RationalQuadraticRBFPolynomials,
}

# In[] Frame definition
class PerturbedFrame(nn.Module):
    def __init__(self, K, noise_type="imqrbf", polynomials="gaussianrbf", shape_type="uniform"):
        super(PerturbedFrame, self).__init__()
        self.K = K  
        self.hk = POLYNOMIALS[polynomials](K, shape_type=shape_type)  
        if noise_type == "imqrbf":
            self.ek = POLYNOMIALS[noise_type](K, shape_type=shape_type)
            self.theta = nn.Parameter(torch.tensor(0.0)) 
        else:
            self.ek = NoisePolynomials(K, noise_type=noise_type)
            self.theta = None

    @property
    def integrals(self):
        return self.weight_sum(self.hk.integrals, self.ek.integrals)   # Shape [1, K]
    
    @property
    def weight(self):
        B1 = math.sqrt(math.pi * 0.5) * self.hk.sigma.sum()  # Gaussian upper bound
        B2 = math.pi * (1.0/self.ek.sigma).sum() 
        R = torch.sqrt(2*B2 / (B1 + 1e-8))   # Add epsilon for numerical stability
        alpha_min = R / (1 + R + 1e-8)      # Lower bound for alpha
        
        # offset = torch.sigmoid(self.theta)  # Map theta to (0,1)
        alpha = alpha_min + (1 - alpha_min) * torch.sigmoid(self.theta)  # alpha ∈ (alpha_min, 1)
        return alpha.squeeze()
   
    def weight_sum(self, hk, ek):
        if self.theta is None:
            return hk + ek
        else:
            weight = self.weight
            return weight * hk + (1 - weight) * ek

    def forward(self, t):
        return self.weight_sum(self.hk(t), self.ek(t)).view(-1, self.K)  # Shape [batch_size, K]
    
    def grad_function(self, t):
        return self.weight_sum(self.hk.grad_function(t), self.ek.grad_function(t)).view(-1, self.K)
    
    def set_cached_sigma(self):
        self.hk.cached_sigma = None
        self.ek.cached_sigma = None

# test
if __name__ == "__main__":
    K = 30  # Number of Legendre polynomials
    poly = ChebyshevPolynomials(K)
    epsilon = 1e-5
    
    # In[]
    # Input tensor t
    batch_size = 5
    # t = torch.rand((batch_size, 1))
    t = epsilon + (1 - 2*epsilon) * torch.linspace(0, 1, batch_size).view(-1,1)
    t = t.detach()
    print(t.T)
    
    # print(poly.coefficients)

    # Compute values of the first K Legendre polynomials at t
    values = poly(t)
    print(f"Values of {poly.__class__.__name__} at t:")
    print(values)

    # Print integrals
    print(f"Integrals of {poly.__class__.__name__} on [0, 1]:")
    print(poly.integrals)
    
    print(f"Derivative of {poly.__class__.__name__} on [0, 1]:")
    print(poly.grad_function(t))
    
    # In[] plot
    import matplotlib.pyplot as plt
    t = torch.linspace(0, 1, 100).view(-1,1)
    values = poly(t)
    plt.figure(figsize=(10, 6))
    for k in range(K):
        plt.plot(t.detach().numpy(), values[:, k].detach().numpy(), label=f"$k = {k}$")
    
    # Add labels and legend
    plt.xlabel("t")
    plt.ylabel("Pk(t)")
    plt.title(f"Transformed {poly.__class__.__name__} on $[0, 1]$ ($K = {K}$)")
    # plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.grid(True)
    plt.show()