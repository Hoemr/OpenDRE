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
        Perform parameter grid search and single-fold cross-validation for KuLSIF.
        Search over lambda (reg_param).
        """
        print("\n--- Starting KuLSIF Parameter Grid Search ---")

        # 1. Split into training and validation sets (one-fold CV)
        X = np.vstack([all_qx, all_px])
        # Target labels (y=1: p1, y=0: p0)
        y = np.hstack([np.zeros(len(all_qx)), np.ones(len(all_px))]) 
        
        # Split data into 80% training and 20% validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Separate p0 and p1 samples from X_train
        qx_train = X_train[y_train == 0]
        px_train = X_train[y_train == 1]
        
        # Define centers for kernel computation
        # Alpha is always computed relative to X_train, so centers must be X_train
        X_centers = X_train
        
        # 2. Define search grid
        lambda_candidates = 10.0 ** (-np.arange(start=-3.0, step=1.0, stop=5.0)) 
        
        best_score = -np.inf
        best_lambda = self.reg_param  # initial value
        gamma = self.rbf_gamma_value

        # 3. Evaluate each candidate
        for reg_param in lambda_candidates:
            try:
                # Temporarily train alpha using X_centers (X_train)
                alpha, cond_num = self._train_kulsif_temp(qx_train, px_train, gamma, reg_param)
                if alpha is None: continue

                # Evaluation metric: maximize E_{p1}[r(x)]
                # Centers must be X_centers (X_train)
                K_val = rbf_kernel(X_val[y_val==1], X_centers, gamma=gamma)
                ratios_p1_val = K_val @ alpha
                
                # Use mean ratio on validation p1 samples as the score
                val_score = np.mean(ratios_p1_val) 

                if val_score > best_score:
                    best_score = val_score
                    best_lambda = reg_param
                    
                print(f"  Lambda={reg_param:.4e}: R(p1)_mean={val_score:.4f}, Cond={cond_num:.2e}")

            except Exception as e:
                import traceback
                print(f"  Lambda={reg_param:.4e}: Failed (Error: {type(e).__name__})")
                continue

        self.reg_param = best_lambda
        print(f"--- Search Complete: Best Lambda={best_lambda:.4e}, R(p1)_mean={best_score:.4f} ---")

    def _train_kulsif_temp(self, qx, px, gamma, reg_param):
        """Temporary KuLSIF training function used during grid search."""
        n0, n1 = len(qx), len(px)
        # X serves as centers: qx + px
        X = np.vstack([qx, px]) 
        
        K = rbf_kernel(X, X, gamma=gamma)
        K_p0 = K[:n0, :]
        
        # Target vector h uses kernel mean over p1 samples
        h = np.mean(K[n0:, :], axis=0) 
        
        # Hessian matrix
        H = K_p0.T @ K_p0 / n0
        
        cond_num = np.linalg.cond(H)
        
        H_reg = H + reg_param * np.eye(len(X))
        
        try:
            alpha = np.linalg.solve(H_reg, h)
        except np.linalg.LinAlgError:
            # Fallback to least squares if direct solve fails
            alpha = np.linalg.lstsq(H_reg, h, rcond=1e-5)[0]
        
        alpha = np.maximum(alpha, 0)
        
        # Normalize so that E_{p0}[r(x)] = 1
        mean_ratio_p0 = np.mean(K_p0 @ alpha)
        if mean_ratio_p0 > 1e-6:
            alpha = alpha / mean_ratio_p0
            return alpha, cond_num
        
        return None, cond_num  # Training failed

    def _train_kulsif(self, qx, px, reg_param, gamma):
        """KuLSIF method – train final model using best parameters."""
        n0, n1 = len(qx), len(px)
        self.training_points = X = np.vstack([qx, px])
        
        K = rbf_kernel(X, X, gamma=gamma)
        K_p0 = K[:n0, :]
        
        h = np.mean(K[n0:, :], axis=0)
        H = K_p0.T @ K_p0 / n0
        
        cond_num = np.linalg.cond(H)
        print(f"H condition number (Final): {cond_num:.2e}")
        
        final_reg = reg_param
        
        # If condition number is too high, enforce stronger regularization for stability
        if cond_num > 1e12:  # 1e12 is a threshold for severe ill-conditioning
            safe_reg = max(reg_param, 100.0)
            if cond_num > 1e14:
                safe_reg = max(safe_reg, 500.0)
            
            final_reg = safe_reg
            print(f"Applying strong regularization due to cond_num. Final Reg={final_reg:.4f}")
        
        H_reg = H + final_reg * np.eye(len(X))
        
        try:
            self.alpha = np.linalg.solve(H_reg, h)
        except np.linalg.LinAlgError:
            self.alpha = np.linalg.lstsq(H_reg, h, rcond=1e-5)[0]
        
        self.alpha = np.maximum(self.alpha, 0)
        
        mean_ratio_p0 = np.mean(K_p0 @ self.alpha)
        if mean_ratio_p0 > 1e-6:
            self.alpha = self.alpha / mean_ratio_p0
            print(f"Normalized alpha by factor {mean_ratio_p0:.4f}")
        else:
            print("Warning: Final KuLSIF normalization failed.")
            self.alpha = None
    
    def _train_kliep(self, qx, px, gamma):
        """KLIEP method – train using fixed gamma."""
        n0, n1 = len(qx), len(px)
        self.training_points = X = np.vstack([qx, px])
        
        K = rbf_kernel(X, X, gamma=gamma)
        
        K_p0 = K[:n0, :]
        K_p1_train = K[n0:, :] 
        
        alpha = np.ones(K.shape[1]) / K.shape[1]
        
        # Increase iterations to ensure convergence
        for iteration in range(1000): 
            r_p1 = K_p1_train @ alpha
            r_p1 = np.maximum(r_p1, 1e-10)
            
            grad = np.mean(K_p1_train / r_p1[:, np.newaxis], axis=0)
            alpha_new = alpha * grad
            
            norm = np.mean(K_p0 @ alpha_new)
            if norm > 1e-8:
                alpha_new = alpha_new / norm
            else:
                break
            
            if np.max(np.abs(alpha_new - alpha)) < 1e-6:
                print(f"KLIEP converged at iteration {iteration}")
                break
            
            alpha = alpha_new
            
            if iteration % 200 == 0:
                r_p0_mean = np.mean(K_p0 @ alpha)
                r_p1_mean = np.mean(K_p1_train @ alpha)
                print(f"KLIEP Iter {iteration}: R(p0)={r_p0_mean:.4f}, R(p1)={r_p1_mean:.4f}, alpha_norm={np.linalg.norm(alpha):.4f}")
        
        self.alpha = alpha
    
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
