# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist

# In[] Kernel-based density ratio estimation
class KernelDensityRatioModel(nn.Module):
    """
    Parametric density ratio estimation model.
    """
    def __init__(self, args, path_model=None, method='kulsif'):
        super().__init__()
        self.args = args
        self.method = method
        self.reg_param = getattr(args, 'reg_param', 1.0)
        self.path_model = path_model
        # Increase the upper limit on the number of training samples
        self.max_samples = getattr(args, 'max_parametric_samples', 2000)
        # Number of kernel centers (drawn from p1/numerator samples, per standard formulation)
        self.n_centers = getattr(args, 'n_centers', 200)
        # KLIEP-specific parameters
        self.kliep_epsilon = getattr(args, 'kliep_epsilon', 1e-4)
        self.kliep_max_iter = getattr(args, 'kliep_max_iter', 5000) 
        
        self.is_trained = False
        self.training_data = {'qx': [], 'px': []}
        self.epoch_count = 0
        
        # Model parameters
        self.alpha = None
        self.training_points = None
        self.rbf_gamma_value = None
        
        # Data normalization parameters
        self.data_mean = None
        self.data_std = None
        
        self.dummy_param = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        
    def forward(self, x):
        """Predict density ratio r(x) = p1(x)/p0(x)."""
        if not self.is_trained or self.alpha is None:
            return torch.ones(x.shape[0], device=x.device)
        
        x_np = x.detach().cpu().numpy()
        
        if self.data_mean is not None:
            x_np = (x_np - self.data_mean) / (self.data_std + 1e-8)
        
        predictions = self._predict_batchwise(x_np)
        predictions = np.clip(predictions, 1e-6, 1e4)
        
        result = torch.from_numpy(predictions).float().to(x.device)
        
        if not hasattr(self, '_debug_printed'):
            print(f"[Forward] Predictions: min={predictions.min():.4f}, max={predictions.max():.4f}, mean={predictions.mean():.4f}")
            self._debug_printed = True
        
        return result
    
    def collect_data(self, qx, px):
        """
        Collect training data – continuously accumulate, but limit total number of batches.
        """
        qx_np = qx.detach().cpu().numpy()
        px_np = px.detach().cpu().numpy()
        
        max_per_batch = 300
        if len(qx_np) > max_per_batch:
            idx = np.random.choice(len(qx_np), max_per_batch, replace=False)
            qx_np = qx_np[idx]
        if len(px_np) > max_per_batch:
            idx = np.random.choice(len(px_np), max_per_batch, replace=False)
            px_np = px_np[idx]

        self.training_data['qx'].append(qx_np)
        self.training_data['px'].append(px_np)

        MAX_TOTAL_BATCHES = 20  # Limit to at most 20 batches
        if len(self.training_data['qx']) > MAX_TOTAL_BATCHES:
            self.training_data['qx'] = self.training_data['qx'][-MAX_TOTAL_BATCHES:]
            self.training_data['px'] = self.training_data['px'][-MAX_TOTAL_BATCHES:]

        total = sum(len(arr) for arr in self.training_data['qx']) + \
                sum(len(arr) for arr in self.training_data['px'])
        
        if len(self.training_data['qx']) == 1:
            print(f"Collected data: {len(qx_np)} p0, {len(px_np)} p1 samples. Total: {total}")
    
    def train_model(self):
        """Train the density ratio model – perform KuLSIF parameter search only on first training."""
        self.epoch_count += 1
        
        if len(self.training_data['qx']) == 0 or len(self.training_data['px']) == 0:
            print(f"Warning: No training data available at epoch {self.epoch_count}")
            return
        
        all_qx = np.vstack(self.training_data['qx'])
        all_px = np.vstack(self.training_data['px'])
        
        print(f"\n{'='*60}")
        print(f"Training Epoch {self.epoch_count} - Method: {self.method}")
        
        # 1. Data normalization and subsampling (unchanged)
        X_all = np.vstack([all_qx, all_px])
        self.data_mean = np.mean(X_all, axis=0)
        self.data_std = np.std(X_all, axis=0) + 1e-8
        
        all_qx = (all_qx - self.data_mean) / self.data_std
        all_px = (all_px - self.data_mean) / self.data_std
        
        n_target = min(self.max_samples, len(all_qx), len(all_px))
        
        # Ensure equal number of p0 and p1 samples
        np.random.shuffle(all_qx)
        np.random.shuffle(all_px)
        all_qx = all_qx[:n_target]
        all_px = all_px[:n_target]
        
        print(f"Training with {len(all_qx)} p0 and {len(all_px)} p1 samples")

        # 2. Determine RBF gamma (run only once)
        if self.rbf_gamma_value is None:
            self.rbf_gamma_value = self._compute_rbf_gamma(all_qx, all_px)
        
        # 3. KuLSIF parameter grid search (run only once and only if method is 'kulsif')
        if self.method == 'kulsif' and self.epoch_count == 1:
            self._perform_kulsif_grid_search(all_qx, all_px)
            # After search, self.reg_param is updated to the best value

        print(f"RBF gamma: {self.rbf_gamma_value:.6f}, Regularization: {self.reg_param:.4f}")
        
        try:
            if self.method == 'kulsif':
                self._train_kulsif(all_qx, all_px, self.reg_param, self.rbf_gamma_value)
            elif self.method == 'kliep':
                self._train_kliep(all_qx, all_px, self.rbf_gamma_value)
            
            if self.alpha is not None:
                # Validate on training set
                test_qx = self._predict_batchwise(all_qx[:100])
                test_px = self._predict_batchwise(all_px[:100])
                
                print(f"Ratios on p0: mean={test_qx.mean():.4f} (should be approx 1)")
                print(f"Ratios on p1: mean={test_px.mean():.4f} (should be >1)")
                print(f"Alpha: min={self.alpha.min():.4f}, max={self.alpha.max():.4f}, "
                      f"mean={self.alpha.mean():.4f}, norm={np.linalg.norm(self.alpha):.4f}")
                
                self.is_trained = True
                print("Training successful!")
                
                if hasattr(self, '_debug_printed'):
                    delattr(self, '_debug_printed')
            else:
                print("Training failed: alpha is None")
                
        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"{'='*60}\n")
    
    def _compute_rbf_gamma(self, qx, px):
        """Compute RBF gamma using the median heuristic: 1 / median^2."""
        n_samples = min(500, len(qx), len(px))
        qx_sample = qx[np.random.choice(len(qx), n_samples, replace=False)]
        px_sample = px[np.random.choice(len(px), n_samples, replace=False)]
        X = np.vstack([qx_sample, px_sample])
        
        distances = pdist(X)
        median_dist = np.median(distances)
        
        if median_dist < 1e-6:
            return 1.0
        
        # Standard heuristic: 1 / (median_dist^2)
        gamma = 1.0 / (median_dist ** 2) 
        gamma = np.clip(gamma, 0.001, 10.0)  # Cap gamma to avoid excessively large values
        
        return gamma
        
    def _perform_kulsif_grid_search(self, all_qx, all_px):
        """
        Perform parameter grid search with cross-validation for KuLSIF.
        Uses the proper uLSIF objective (0.5*E_q[r²] - E_p[r]) on validation data
        to select the regularization parameter lambda.
        """
        print("\n--- Starting KuLSIF Parameter Grid Search ---")

        # 1. Split into training and validation sets (stratified 80/20)
        X = np.vstack([all_qx, all_px])
        y = np.hstack([np.zeros(len(all_qx)), np.ones(len(all_px))])

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        qx_train = X_train[y_train == 0]
        px_train = X_train[y_train == 1]
        qx_val = X_val[y_val == 0]
        px_val = X_val[y_val == 1]

        # 2. Pre-select centers from px_train (numerator samples) — fixed for fair comparison
        n_ctr = min(self.n_centers, len(px_train))
        center_idx = np.random.choice(len(px_train), n_ctr, replace=False)
        centers_train = px_train[center_idx]

        # 3. Define search grid over lambda (log-spaced)
        lambda_candidates = np.logspace(-6, 1, 15)

        best_loss = np.inf
        best_lambda = self.reg_param
        gamma = self.rbf_gamma_value

        # 4. Evaluate each lambda candidate
        for reg_param in lambda_candidates:
            try:
                # Train alpha on training set with pre-selected centers
                alpha, cond_num = self._train_kulsif_temp(
                    qx_train, px_train, gamma, reg_param, centers=centers_train
                )
                if alpha is None:
                    continue

                # Evaluate proper uLSIF loss on validation set:
                #   J(r) = 0.5 * E_q[r²] - E_p[r]
                # Lower is better.
                K_val_q = rbf_kernel(qx_val, centers_train, gamma=gamma)
                K_val_p = rbf_kernel(px_val, centers_train, gamma=gamma)
                ratios_q_val = K_val_q @ alpha
                ratios_p_val = K_val_p @ alpha
                val_loss = 0.5 * np.mean(ratios_q_val ** 2) - np.mean(ratios_p_val)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_lambda = reg_param

                print(f"  Lambda={reg_param:.4e}: Loss={val_loss:.4f}, "
                      f"R(q)_mean={np.mean(ratios_q_val):.3f}, "
                      f"R(p)_mean={np.mean(ratios_p_val):.3f}, Cond={cond_num:.2e}")

            except Exception as e:
                print(f"  Lambda={reg_param:.4e}: Failed (Error: {type(e).__name__})")
                continue

        self.reg_param = best_lambda
        print(f"--- Search Complete: Best Lambda={best_lambda:.4e}, Val Loss={best_loss:.4f} ---")

    def _train_kulsif_temp(self, qx, px, gamma, reg_param, centers=None):
        """Temporary KuLSIF training function used during grid search.

        Per the standard uLSIF formulation (Kanamori et al. 2009):
          r(x) = Σ α_i k(x, c_i)   with centers c_i drawn from px (numerator).
          Ĥ = (1/n_q) Σ_{x~q} φ(x) φ(x)ᵀ      [n_ctr × n_ctr]
          ĥ = (1/n_p) Σ_{x~p} φ(x)             [n_ctr]
          α = (Ĥ + λ I)⁻¹ ĥ
        """
        n_q = len(qx)

        if centers is None:
            n_ctr = min(self.n_centers, len(px))
            idx = np.random.choice(len(px), n_ctr, replace=False)
            centers = px[idx]
        n_ctr = len(centers)

        # Kernel matrices: rows=samples, cols=centers
        K_q = rbf_kernel(qx, centers, gamma=gamma)  # (n_q, n_ctr)
        K_p = rbf_kernel(px, centers, gamma=gamma)  # (n_p, n_ctr)

        # h = E_p[φ(x)]  — mean kernel embedding of numerator distribution
        h = np.mean(K_p, axis=0)                     # (n_ctr,)

        # H = E_q[φ(x) φ(x)ᵀ] — covariance of kernel features under denominator
        H = K_q.T @ K_q / n_q                        # (n_ctr, n_ctr)

        cond_num = np.linalg.cond(H + reg_param * np.eye(n_ctr))

        H_reg = H + reg_param * np.eye(n_ctr)

        try:
            alpha = np.linalg.solve(H_reg, h)
        except np.linalg.LinAlgError:
            alpha = np.linalg.lstsq(H_reg, h, rcond=1e-5)[0]

        # Non-negativity truncation (uLSIF is "unconstrained", but truncation is standard post-processing)
        alpha = np.maximum(alpha, 0)

        # Normalize so that E_q[r(x)] = 1
        mean_ratio_q = np.mean(K_q @ alpha)
        if mean_ratio_q > 1e-6:
            alpha = alpha / mean_ratio_q
            return alpha, cond_num

        return None, cond_num

    def _train_kulsif(self, qx, px, reg_param, gamma):
        """KuLSIF method — train final model with best regularization parameter.

        Per the standard uLSIF formulation (Kanamori et al. 2009):
          Centers are drawn from px (numerator) only.
          α = (Ĥ + λ I)⁻¹ ĥ,  with Ĥ from q-samples, ĥ from p-samples.
        """
        n_q, n_p = len(qx), len(px)

        # Select centers from px (numerator) samples only
        n_ctr = min(self.n_centers, n_p)
        ctr_idx = np.random.choice(n_p, n_ctr, replace=False)
        self.training_points = px[ctr_idx]

        # Kernel matrices: rows=samples, cols=centers
        K_q = rbf_kernel(qx, self.training_points, gamma=gamma)  # (n_q, n_ctr)
        K_p = rbf_kernel(px, self.training_points, gamma=gamma)  # (n_p, n_ctr)

        # h = E_p[φ(x)]
        h = np.mean(K_p, axis=0)                                   # (n_ctr,)
        # H = E_q[φ(x) φ(x)ᵀ]
        H = K_q.T @ K_q / n_q                                      # (n_ctr, n_ctr)

        cond_num = np.linalg.cond(H)
        print(f"H matrix ({n_ctr}×{n_ctr}), condition number (unregularized): {cond_num:.2e}")

        final_reg = reg_param

        # If condition number is extremely high, increase regularization
        if cond_num > 1e10:
            final_reg = max(reg_param, 1.0)
            print(f"Increasing regularization: {final_reg:.4f}")

        H_reg = H + final_reg * np.eye(n_ctr)

        try:
            self.alpha = np.linalg.solve(H_reg, h)
        except np.linalg.LinAlgError:
            self.alpha = np.linalg.lstsq(H_reg, h, rcond=1e-5)[0]

        # Non-negativity truncation
        self.alpha = np.maximum(self.alpha, 0)

        # Normalize so that E_q[r(x)] = 1
        mean_ratio_q = np.mean(K_q @ self.alpha)
        if mean_ratio_q > 1e-6:
            self.alpha = self.alpha / mean_ratio_q
            print(f"Normalized alpha by E_q[r] = {mean_ratio_q:.4f}")
        else:
            print("Warning: Final KuLSIF normalization failed (E_q[r] ≈ 0).")
            self.alpha = None
    
    def _train_kliep(self, qx, px, gamma):
        """KLIEP method — train with additive gradient ascent.

        Standard KLIEP (Sugiyama et al. 2008):
          Centers c_i are drawn from px (test/numerator) only.
          Maximise  J(α) = (1/n_p) Σ_j log ŵ(x_j^p)
          subject to (1/n_q) Σ_i ŵ(x_i^q) = 1, α ≥ 0.

        Each iteration:
          1. Additive gradient step:  α += ε · Aᵀ · (1 / (Aα))
          2. Mean adjustment:          α += b · ((1 − bᵀα) / (bᵀb))
          3. Non-negativity:           α = max(0, α)
          4. Re-normalisation:         α = α / (bᵀα)
        """
        n_q, n_p = len(qx), len(px)

        # Select centers from px (numerator / test) samples only
        n_ctr = min(self.n_centers, n_p)
        ctr_idx = np.random.choice(n_p, n_ctr, replace=False)
        self.training_points = px[ctr_idx]

        # A: kernel of test (p1) samples against centers  — shape (n_p, n_ctr)
        # b: mean kernel of train (p0) samples against centers — shape (n_ctr, 1)
        A = rbf_kernel(px, self.training_points, gamma=gamma)
        b = np.mean(rbf_kernel(qx, self.training_points, gamma=gamma), axis=0)
        b = b.reshape(n_ctr, 1)

        # Initialise alpha as uniform positive vector
        alpha = np.ones((n_ctr, 1)) / n_ctr
        epsilon = self.kliep_epsilon
        converged = False

        for iteration in range(self.kliep_max_iter):
            # 1. Compute ŵ(x^p) = A @ α
            w_p = A @ alpha                     # (n_p, 1)
            w_p = np.maximum(w_p, 1e-10)        # numerical stability

            # 2. Additive gradient-ascent step
            grad = A.T @ (1.0 / w_p)            # (n_ctr, 1)
            alpha_new = alpha + epsilon * grad

            # 3. Mean adjustment: enforce bᵀα = 1 in least-squares sense
            b_dot_alpha = b.T @ alpha_new       # scalar
            adjustment = b * ((1.0 - b_dot_alpha) / (b.T @ b))
            alpha_new = alpha_new + adjustment

            # 4. Project onto non-negative orthant
            alpha_new = np.maximum(0, alpha_new)

            # 5. Re-normalise
            b_dot_new = b.T @ alpha_new
            if b_dot_new < 1e-10:
                print("KLIEP: degenerated (bᵀα ≈ 0), restarting with uniform α.")
                alpha = np.ones((n_ctr, 1)) / n_ctr
                epsilon = max(epsilon * 0.1, 1e-8)
                continue
            alpha_new = alpha_new / b_dot_new

            # Check convergence
            max_diff = np.max(np.abs(alpha_new - alpha))
            alpha = alpha_new

            if max_diff < 1e-6:
                converged = True
                print(f"KLIEP converged at iteration {iteration}")
                break

            if iteration % 500 == 0:
                r_q_mean = np.mean(rbf_kernel(qx, self.training_points, gamma=gamma) @ alpha)
                r_p_mean = np.mean(w_p)
                print(f"KLIEP Iter {iteration}: R(q)={r_q_mean:.4f}, R(p)={r_p_mean:.4f}, "
                      f"|α|={np.linalg.norm(alpha):.4f}")

        if not converged:
            print(f"KLIEP: max iterations ({self.kliep_max_iter}) reached without convergence.")

        # Store alpha as 1D array for consistency with predict
        self.alpha = alpha.ravel()
    
    def _predict_batchwise(self, x, batch_size=1000):
        """Make predictions in batches."""
        if self.alpha is None or self.training_points is None:
            return np.ones(len(x))
        
        n = len(x)
        predictions = np.zeros(n)
        
        gamma = self.rbf_gamma_value  # Use gamma determined during training
        
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            x_batch = x[i:end_i]
            
            K = rbf_kernel(x_batch, self.training_points, gamma=gamma)
            predictions[i:end_i] = K @ self.alpha
        
        return predictions
