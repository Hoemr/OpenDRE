# -*- coding: utf-8 -*-
"""Neural density ratio estimation training step."""
import torch
import torch.nn.functional as F

class NeuralDensityRatioTrainStepFn:
    """
    Training step function for density ratio estimation.
    Estimates f(x) = log r(x), where r(x) = p1(x) / p0(x).
    Methods based on f-Divergence Duality in Classification.
    """

    def __init__(self, args, path_model=None, method='logistic'):
        self.args = args
        self.method = method
        self.reg_param = getattr(args, 'reg_param', 0.1)
        self.path = path_model
        
        # Check if path model is available for NLL calculation (done once at init)
        self.has_path_model = path_model is not None and hasattr(path_model, 'prior_logp')
        
        # Exponential Weight (EW) parameter
        self.ew_alpha = getattr(args, 'ew_alpha', 2.0)  # Default from paper
        
        method = self.method.lower()
        if method == 'kulsif' or method == 'chisq' or method == 'sq': # square loss (sq) is a special case of Chi-Squared
            # Pearson Chi-Squared Loss (uLSIF/KuLSIF equivalent)
            self.loss_fn = self.chisq_loss
        elif method == 'logistic' or method == 'jsd' or method == 'bce':
            # JSD / Standard Binary Classification Loss
            self.loss_fn = self.jsd_bce_loss
        elif method == 'nce':
            # Noise-Contrastive Estimation (Weighted BCE)
            self.loss_fn = self.nce_loss
        elif method == 'infonce':
            # InfoNCE (Contrastive Multi-class Loss)
            self.loss_fn = self.infonce_loss
        elif method == 'rkl':
            self.loss_fn = self.rkl_loss
        elif method == 'hellinger':
            self.loss_fn = self.hellinger_loss
        elif method == 'kl' or method == "exp": 
            self.loss_fn = self.kl_loss
        elif method == 'gamma':
            # Gamma-DRE Loss
            self.loss_fn = self.gamma_dre_loss
        elif method == 'ew':
            # Exponential Weight Loss from ICLR 2025 paper
            self.loss_fn = self.exponential_weight_loss
        elif method == 'pw':   # poly_weight
            # Polynomial Weight Loss from ICLR 2025 paper
            self.loss_fn = self.poly_weight_loss
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        
    def __call__(self, model, batch, step=None):
        qx, px = batch 
        loss, nll = self.loss_fn(model, qx, px) 
        
        return {'loss': loss.mean(), 'nll': nll.mean()}
    
    def _calc_nll(self, log_r_px, px_batch):
        # NLL = - log p1(x) = - (log r(x) + log p0(x))
        # Return 0 if SDE is not available (checked once at init)
        if not self.has_path_model:
            return torch.zeros_like(log_r_px)
        return - (log_r_px + self.path.prior_logp(px_batch))
    
    def compute_weight_function(self, x, tau=50.0):
        """
        Weight function from the paper: w(x) = exp(-||x||^4_4 / tau)
        
        Implemented according to paper Example 3.2:
        - L4 norm: ||x||^4_4 = sum_{i=1}^d x_i^4
        - tau: temperature parameter controlling weight decay rate
        
        Args:
            x: input data [batch_size, dim]
            tau: temperature parameter (50 in the paper)
        
        Returns:
            w(x): weight values [batch_size, 1]
        """
        # Compute L4 norm: ||x||^4_4 = sum(x_i^4)
        l4_norm = torch.sum(x**4, dim=1, keepdim=True)
        
        # Weight function: w(x) = exp(-||x||^4_4 / tau)
        weights = torch.exp(-l4_norm / tau)
        
        return weights.detach()
    
    def gamma_dre_loss(self, model, qx, px, gamma=0.01):
        """
        Gamma-DRE Loss Function 
        Loss = -1/γ * log(E_p[exp(γ * f(x))]) + 1/(1+γ) * log(E_q[exp((1+γ) * f(x))])
        
        where f(x) = log r(x) is the density ratio function.
        
        Reference:
        [1] Nagumo, R., & Fujisawa, H. (2024). Density Ratio Estimation with Doubly Strong Robustness.
            Proceedings of the 41st International Conference on Machine Learning.
        """
        log_r_qx = model(qx)#.clamp(min=-30.0, max=30.0)  # f(x) for x ~ p0
        log_r_px = model(px)#.clamp(min=-30.0, max=30.0)  # f(x) for x ~ p1
        
        # w(x) = exp(-||x||^4_4 / τ)
        w_qx = self.compute_weight_function(qx, tau=50.0)  
        w_px = self.compute_weight_function(px, tau=50.0)
        
        # the first term: -1/γ * log(E_p[w(x) * exp(γ * f(x))])
        gamma_log_p = gamma * log_r_px
        m1 = torch.max(gamma_log_p.detach())
        stable_exp_p = torch.exp(gamma_log_p - m1)
        weighted_exp_p = w_px * stable_exp_p
        log_mean_exp_p = torch.log(torch.mean(weighted_exp_p) + 1e-12) + m1
        term1 = - (1.0 / gamma) * log_mean_exp_p
        
        # the second term: 1/(1+γ) * log(E_q[w(x) * exp((1+γ) * f(x))])
        gamma1_log_q = (1.0 + gamma) * log_r_qx
        m2 = torch.max(gamma1_log_q.detach())
        stable_exp_q = torch.exp(gamma1_log_q - m2)
        weighted_exp_q = w_qx * stable_exp_q
        log_mean_exp_q = torch.log(torch.mean(weighted_exp_q) + 1e-12) + m2
        term2 = (1.0 / (1.0 + gamma)) * log_mean_exp_q

        loss = term1 + term2
        
        nll = self._calc_nll(log_r_px, px)
        return loss, nll
    
    def jsd_bce_loss(self, model, qx, px):
        """
        JSD Loss (Equivalent to standard Binary Cross-Entropy / NS-GAN).
        This is the standard DRE loss derived from classification, also known as Logistic Loss.
        Equivalent to training a classifier to distinguish p0 vs p1.
        It is also equivalent to NCE with K=1.
        Loss = E_{p0}[softplus(f)] + E_{p1}[softplus(-f)]
        
        Reference:
        [2] Nowozin, S., Cseke, B., & Tomioka, R. (2016). f-GAN: Training generative neural 
            samplers using variational divergence minimization. NeurIPS.
        """
        log_r_qx = model(qx) # f(x) for x ~ p0
        log_r_px = model(px) # f(x) for x ~ p1
        
        # Classification loss (BCE)
        loss_q = torch.mean(F.softplus(log_r_qx))    # E_{p0}[log(1 + r)]
        loss_p = torch.mean(F.softplus(-log_r_px))   # E_{p1}[log(1 + 1/r)]
        
        loss = loss_q + loss_p
        nll = self._calc_nll(log_r_px, px)
        return loss, nll
    
    def nce_loss(self, model, qx, px):
        """
        Noise-Contrastive Estimation (NCE) Loss.

        Gutmann & Hyvärinen (2010, 2012).  Given K = M/N noise samples per
        positive sample, the classifier is  g(x) = log r(x) - log K  and the
        correctly weighted NCE objective is

            L = (1/N) [ \Sigma_{x~p} softplus(-g(x)) + \Sigma_{x~q} softplus(g(x)) ].

        Reference:
        [3] Gutmann, M., & Hyvärinen, A. (2010). Noise-contrastive estimation:
            A new estimation principle for unnormalized statistical models. AISTATS.
        """
        N = px.shape[0]
        M = qx.shape[0]
        K = M / N
        log_k = torch.log(torch.tensor(K, device=px.device, dtype=torch.float32))
        log_r_qx = model(qx)  # f(x) for x ~ p0
        log_r_px = model(px)  # f(x) for x ~ p1
        # Classifier logits: g(x) = f(x) - log K
        g_qx = log_r_qx - log_k
        g_px = log_r_px - log_k
        # Correct NCE weighting: sum over all samples, divide by N
        loss = (torch.sum(F.softplus(-g_px)) + torch.sum(F.softplus(g_qx))) / N

        nll = self._calc_nll(log_r_px, px)
        return loss, nll
    
    def infonce_loss(self, model, qx, px):
        """
        InfoNCE Loss (Contrastive Learning Loss).
        [Oord et al., 2018]
        Treats DRE as a multi-class classification problem.
        For each p1 sample, it must be identified from a pool of p0 samples.
        
        Reference:
        [4] van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation learning with 
            contrastive predictive coding. arXiv preprint arXiv:1807.03748.
        """
        N = px.shape[0] # Number of positive samples
        M = qx.shape[0] # Number of negative samples
        log_r_px = model(px) # [N, 1]
        log_r_qx = model(qx) # [M, 1]
        # We compute N separate contrastive losses
        
        # [N, 1] -> [N]
        pos_logits = log_r_px.squeeze(-1)
        # [M, 1] -> [M]
        neg_logits = log_r_qx.squeeze(-1)
        # Create the full logit matrix: [N, M + 1]
        # Each row i corresponds to positive sample px_i
        # Column 0 is the positive logit, Columns 1...M are negative logits
        
        # Expand positive logits to [N, 1]
        pos_logits_expanded = pos_logits.view(-1, 1)
        # Expand negative logits to [N, M] (all positives share all negatives)
        neg_logits_expanded = neg_logits.expand(N, -1) 
        # Combine: [N, 1 + M]
        logits = torch.cat([pos_logits_expanded, neg_logits_expanded], dim=1)
        # Labels are always 0 (the first column is the positive sample)
        labels = torch.zeros(N, dtype=torch.long, device=px.device)
        # Standard Cross-Entropy Loss
        loss = F.cross_entropy(logits, labels)
        nll = self._calc_nll(log_r_px, px)
        return loss, nll
    
    def kl_loss(self, model, qx, px):
        """
        Forward KL density-ratio estimation (Nguyen et al., 2007).

        The Fenchel dual for KL(p1 || p0) is:
            D_KL = sup_T  E_{p1}[T(x)] - E_{p0}[exp(T(x) - 1)]

        The maximiser satisfies  T*(x) = 1 + log(p1(x)/p0(x)).
        Since the model outputs  log r(x) = log(p1/p0), we set
            T(x) = log r(x) + 1

        Substituting into the dual and dropping the constant -1:
            loss = -E_{p1}[log r] + E_{p0}[r]

        Reference:
        [5] Nguyen, X., Wainwright, M. J., & Jordan, M. I. (2007). Estimating divergence
            functionals and the likelihood ratio by penalized convex risk minimization. NeurIPS.
        """
        log_r_px = model(px)  # log r(x), x ~ p1
        log_r_qx = model(qx)  # log r(x), x ~ p0

        log_r_qx = torch.clamp(log_r_qx, min=-10, max=10)
        r_qx = torch.exp(log_r_qx)

        # loss = -E_p1[log r] + E_p0[r]   (constant -1 dropped; gradient-preserving)
        loss = -torch.mean(log_r_px) + torch.mean(r_qx)

        nll = self._calc_nll(log_r_px, px)
        return loss, nll
    
    def rkl_loss(self, model, qx, px):
        """
        Reverse KL Loss (KL(p0 || p1) variant).
        f(r) = - log r. Dual form: E_{p0}[-f] - E_{p1}[e^{-f}]
        
        Reference:
        [6] Sugiyama, M., Suzuki, T., & Kanamori, T. (2012). Density-ratio matching under the 
            Bregman divergence: a unified framework of density-ratio estimation. Annals of ISM.
        """
        log_r_qx = model(qx)
        log_r_px = model(px)
        
        # 1/r(x) for x ~ p1
        r_inv_px = torch.exp(torch.clamp(-log_r_px, min=-10, max=10)) 
        
        loss = -torch.mean(log_r_qx) - torch.mean(r_inv_px)
        
        # Optional L2 penalty on outputs
        # loss += self.reg_param * torch.mean(log_r_qx**2)
        nll = self._calc_nll(log_r_px, px)
        return loss, nll
    
    def chisq_loss(self, model, qx, px):
        """
        Pearson Chi-Squared Loss (uLSIF/KuLSIF equivalent).
        f(r) = 0.5 * (r - 1)^2. Dual form: 0.5 * E_{p0}[r^2] - E_{p1}[r]
        (Note: We use f = log r)
        
        Reference:
        [7] Kanamori, T., Hido, S., & Sugiyama, M. (2009). A least-squares approach to 
            direct importance estimation. Journal of Machine Learning Research.
        """
        log_r_qx = model(qx)
        log_r_px = model(px)
        
        r_qx = torch.exp(torch.clamp(log_r_qx, min=-5., max=5.))
        r_px = torch.exp(torch.clamp(log_r_px, min=-5., max=5.))
        
        loss = 0.5 * torch.mean(r_qx**2) - torch.mean(r_px)
        
        # L2 penalty on outputs (optional, analogous to lambda in KuLSIF)
        loss += self.reg_param * torch.mean(r_qx**2)
        nll = self._calc_nll(log_r_px, px)
        return loss, nll
    
    def hellinger_loss(self, model, qx, px):
        r"""
        Hellinger-distance density-ratio estimation (Nowozin et al., 2016).

        Hellinger f-divergence:  f(u) = (\sqrt{u} - 1)^2,  dom_f* = {t < 1}.

        Using the f-GAN variational form with T = 1 - 1/\sqrt{r}:
            loss = E_q[\sqrt{r}] + E_p[1 / \sqrt{r}]

        Gradient check — at optimum:
            q / \sqrt{r}  =  p / r^{3/2}   =>   r(x) = p(x) / q(x).

        Reference:
        [8] Nowozin, S., Cseke, B., & Tomioka, R. (2016). f-GAN: Training generative
            neural samplers using variational divergence minimization. NeurIPS.
        """
        log_r_qx = model(qx)
        log_r_px = model(px)
        r_qx = torch.exp(torch.clamp(log_r_qx, min=-10, max=10))
        r_px = torch.exp(torch.clamp(log_r_px, min=-10, max=10))

        sqrt_r_qx = torch.sqrt(r_qx + 1e-8)
        inv_sqrt_r_px = 1.0 / torch.sqrt(r_px + 1e-8)

        # loss = E_q[\sqrt{r}] + E_p[1 / \sqrt{r}]
        loss = torch.mean(sqrt_r_qx) + torch.mean(inv_sqrt_r_px)

        nll = self._calc_nll(log_r_px, px)
        return loss, nll
    
    def exponential_weight_loss(self, model, qx, px):
        """
        Exponential Weight (EW) Loss [Zellinger, ICLR 2025]
        Prioritizes accurate estimation of large density ratio values.
        
        Loss formulation from paper:
        ℓ(1,y) = -y, ℓ(-1,y) = 0.5 * y * (log(2y) - 1)
        with β(x) = 0.5 * log(2f(x))
        
        Corresponds to Bregman divergence with φ''(c) = e^{2c}
        
        Reference:
        [9] Zellinger, W. (2025). Binary Losses for Density Ratio Estimation.
            ICLR 2025. (Exponential Weight method)
        """
        raw_f_qx = model(qx).clamp(max=100.)  # f(x) for x ~ p0, ∈ R
        raw_f_px = model(px).clamp(max=100.)  # f(x) for x ~ p1, ∈ R
        
        # Map model outputs to positive domain inside the loss function
        # Using softplus: R → (0, +∞)
        f_qx = F.softplus(raw_f_qx) + 1e-8
        f_px = F.softplus(raw_f_px) + 1e-8
        
        # Loss components (based on transformed f(x) > 0)
        loss_p = -torch.mean(f_px)  # ℓ(1,y) = -y
        loss_q = torch.mean(0.5 * f_qx * (torch.log(2 * f_qx) - 1))  # ℓ(-1,y)
        
        # Tikhonov regularization from the paper: α∥f∥², used for stable training
        # tikhonov_reg = 1e-2 * F.softplus(torch.mean(f_px**2) + torch.mean(f_qx**2))  
        tikhonov_reg = 1e-2 * (torch.mean(f_px**2) + torch.mean(f_qx**2))  
        
        loss = loss_q + loss_p + tikhonov_reg 

        # Density ratio estimator: β(x) = 0.5 * log(2 * f(x))
        estimated_log_ratio = 0.5 * torch.log(2 * f_px)
        
        nll = self._calc_nll(estimated_log_ratio, px)
        return loss, nll
    
    def poly_weight_loss(self, model, qx, px, k=1):
        """
        Polynomial Weight Loss [Zellinger, ICLR 2025]
        Family of losses with φ''(c) = c^{2k} for k ∈ {0,1,2,3,...}
        
        For k=0: Equivalent to KuLSIF (Chi-squared loss)
        For k>0: Prioritizes larger density ratio values
        
        Loss: ℓ(1,y) = -y, ℓ(-1,y) = ((1+k)y)^{2+k}/(1+k)/(2+k)
        Estimator: β(x) = ((1+k)f(x))^{1/(1+k)}
        
        Reference:
        [9] Zellinger, W. (2025). Binary Losses for Density Ratio Estimation.
            ICLR 2025. (Polynomial Weight methods)
        """
        # Model outputs f(x) ∈ R
        raw_f_qx = model(qx)  # ∈ R
        raw_f_px = model(px)  # ∈ R
        
        # Map to positive domain
        f_qx = F.softplus(raw_f_qx) + 1e-8
        f_px = F.softplus(raw_f_px) + 1e-8
        
        loss_p = -torch.mean(f_px)
        term_q = ((1 + k) * f_qx) ** (2 + k) / ((1 + k) * (2 + k))
        loss_q = torch.mean(term_q)
        
        tikhonov_reg = 1e-2 * (torch.mean(f_px**2) + torch.mean(f_qx**2))  
        
        loss = loss_p + loss_q + tikhonov_reg
        
        # Density ratio estimator
        estimated_ratio = ((1 + k) * f_px) ** (1 / (1 + k))
        estimated_log_ratio = torch.log(estimated_ratio + 1e-8)
        
        nll = self._calc_nll(estimated_log_ratio, px)
        return loss, nll
    
    def kliep_loss(self, model, qx, px):
        """
        KLIEP-style loss (kernel approximation).
        Minimizes KL(p1 || r * p0) = -E_{p1}[log r(x)] + E_{p0}[r(x)]
        with soft constraint E_{p0}[r(x)] approx 1 via penalty.
        """
        log_r_qx = model(qx)  # log r(x), x ~ p0
        log_r_px = model(px)  # log r(x), x ~ p1

        # Prevent explosion
        log_r_qx = torch.clamp(log_r_qx, min=-10, max=10)
        log_r_px = torch.clamp(log_r_px, min=-10, max=10)

        r_qx = torch.exp(log_r_qx)  # r(x) for x ~ p0

        # KLIEP objective: -E_{p1}[log r] + E_{p0}[r]
        kl_term = -torch.mean(log_r_px) + torch.mean(r_qx)

        # Optional: enforce E_{p0}[r(x)] approx 1 (normalization constraint)
        loss = kl_term #+ self.reg_param * (torch.mean(r_qx) - 1.0) ** 2
        nll = self._calc_nll(log_r_px, px)
        
        return loss, nll
    
    def adaptive_loss_weighting(self, loss_list, temperature=5.0, eps=1e-8):
        """
        This method automatically assigns higher weights to losses with smaller magnitudes,
        effectively balancing different objectives without manually tuning loss coefficients.

        Args:
            loss_list (List[Tensor]): List of per-batch loss tensors, each with shape [B, ...].
            temperature (float): Softmax temperature. Higher values produce more uniform weights,
                                lower values make weighting more "winner-takes-all".
            eps (float): Small constant for numerical stability to avoid division by zero.
        """
        mean_losses = [l.squeeze().mean() for l in loss_list]

        detached_means = torch.stack([ml.detach().abs() for ml in mean_losses], dim=0)
        weights = 1.0 / (detached_means + eps)
        weights = F.softmax(weights / temperature, dim=0)

        total_loss = torch.sum(weights * torch.stack(mean_losses, dim=0))
        return total_loss
