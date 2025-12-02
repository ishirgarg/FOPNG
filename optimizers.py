import torch
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Dict, Any, List, Union
from tqdm import tqdm

from config import Config
from gradients import GradientMemory, GradientCollector, GTLCollector, AVECollector
from fisher import FisherEstimator, DiagonalFisherEstimator, FullFisherEstimator
from utils import get_param_count, apply_update
from gradients import get_grad_vector, set_grad_vector
from logger import log

class ContinualMethod(ABC):
    """Abstract base class for continual learning methods."""
    
    @abstractmethod
    def setup(self, model: nn.Module, config: Config):
        """Initialize method-specific state."""
        pass
    
    @abstractmethod
    def train_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        criterion: nn.Module,
        config: Config,
        task_id: int,
        multihead: bool = False,
        progress_desc: Optional[str] = None
    ) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (loss, accuracy)
        """
        pass
    
    @abstractmethod
    def after_task(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        task_id: int,
        config: Config,
        multihead: bool = False
    ):
        """Called after finishing training on a task."""
        pass
    
    @property
    def name(self) -> str:
        """Return method name for logging."""
        return self.__class__.__name__.replace('Method', '')


class SGDMethod(ContinualMethod):
    """Vanilla SGD baseline (no continual learning)."""
    
    def setup(self, model: nn.Module, config: Config):
        pass
    
    def train_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        criterion: nn.Module,
        config: Config,
        task_id: int,
        multihead: bool = False,
        progress_desc: Optional[str] = None
    ) -> Tuple[float, float]:
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        iterator = tqdm(train_loader, desc=progress_desc, leave=False) if progress_desc else train_loader
        
        for x, y in iterator:
            x = x.to(config.device)
            y = y.to(config.device)
            
            optimizer.zero_grad()
            
            if multihead:
                logits = model(x, task_id=task_id)
            else:
                logits = model(x)
            
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
        
        return total_loss / total_samples, total_correct / total_samples
    
    def after_task(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        task_id: int,
        config: Config,
        multihead: bool = False
    ):
        pass

class AdamMethod(ContinualMethod):
    """Vanilla Adam baseline (no continual learning)."""
    
    def setup(self, model: nn.Module, config: Config):
        pass
    
    def train_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        criterion: nn.Module,
        config: Config,
        task_id: int,
        multihead: bool = False,
        progress_desc: Optional[str] = None
    ) -> Tuple[float, float]:
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        iterator = tqdm(train_loader, desc=progress_desc, leave=False) if progress_desc else train_loader
        
        for x, y in iterator:
            x = x.to(config.device)
            y = y.to(config.device)
            
            optimizer.zero_grad()
            
            if multihead:
                logits = model(x, task_id=task_id)
            else:
                logits = model(x)
            
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
        
        return total_loss / total_samples, total_correct / total_samples
    
    def after_task(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        task_id: int,
        config: Config,
        multihead: bool = False
    ):
        pass


class OGDMethod(ContinualMethod):
    """
    Orthogonal Gradient Descent.
    Projects gradients to be orthogonal to stored directions from previous tasks.
    """
    
    def __init__(
        self,
        collector: GradientCollector = None,
        max_directions: int = 2000
    ):
        self.collector = collector or GTLCollector()
        self.memory = GradientMemory(mode='orthonormal', max_directions=max_directions)
    
    def setup(self, model: nn.Module, config: Config):
        self.memory.clear()
    
    def train_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        criterion: nn.Module,
        config: Config,
        task_id: int,
        multihead: bool = False,
        progress_desc: Optional[str] = None
    ) -> Tuple[float, float]:
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # Accumulators for gradient norms and ratios (log average per epoch)
        raw_grad_norms = []
        proj_grad_norms = []
        proj_to_raw_ratios = []
        projection_relative_changes = []
        
        num_directions = len(self.memory)
        
        iterator = tqdm(train_loader, desc=progress_desc, leave=False) if progress_desc else train_loader
        
        for x, y in iterator:
            x = x.to(config.device)
            y = y.to(config.device)
            
            optimizer.zero_grad()
            
            if multihead:
                logits = model(x, task_id=task_id)
            else:
                logits = model(x)
            
            loss = criterion(logits, y)
            loss.backward()
            
            # Get raw gradient and compute norm
            g = get_grad_vector(model)
            raw_norm = g.norm().item()
            raw_grad_norms.append(raw_norm)
            
            # Project gradient if we have stored directions
            if num_directions > 0:
                g_tilde = self.memory.project_orthogonal(g)
                proj_norm = g_tilde.norm().item()
                proj_grad_norms.append(proj_norm)
                proj_to_raw_ratios.append(proj_norm / (raw_norm + 1e-8))
                
                # Compute relative change from projection
                diff_norm = (g - g_tilde).norm().item()
                relative_change = diff_norm / (raw_norm + 1e-10)
                projection_relative_changes.append(relative_change)
                
                set_grad_vector(model, g_tilde)
            else:
                proj_grad_norms.append(raw_norm)  # No projection, same as raw
                proj_to_raw_ratios.append(1.0)  # Ratio is 1 when no projection
                projection_relative_changes.append(0.0)  # No change when no projection
            
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
        
        # Log comprehensive OGD metrics
        log_metrics = {
            "task_id": task_id,
            "ogd_gradients/raw_grad_norm_mean": np.mean(raw_grad_norms),
            "ogd_gradients/raw_grad_norm_std": np.std(raw_grad_norms),
            "ogd_gradients/projected_grad_norm_mean": np.mean(proj_grad_norms),
            "ogd_gradients/projected_grad_norm_std": np.std(proj_grad_norms),
            "ogd_gradients/projected_to_raw_ratio_mean": np.mean(proj_to_raw_ratios),
            "ogd_gradients/projected_to_raw_ratio_std": np.std(proj_to_raw_ratios),
            "ogd_gradients/projection_relative_change_mean": np.mean(projection_relative_changes),
            "ogd_gradients/projection_relative_change_std": np.std(projection_relative_changes),
            "ogd_gradients/num_directions": num_directions,
        }
        
        log(log_metrics)
        
        return total_loss / total_samples, total_correct / total_samples
    
    def after_task(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        task_id: int,
        config: Config,
        multihead: bool = False
    ):
        print(f"Collecting OGD directions from task {task_id}...")
        self.collector.collect(
            self.memory,
            model,
            train_loader,
            config.grads_per_task,
            config.device,
            multihead=multihead,
            task_id=task_id if multihead else None
        )


class FOPNGMethod(ContinualMethod):
    """
    Fisher-Orthogonal Projected Natural Gradient.
    Uses Fisher information to define a Riemannian metric for projection.
    """
    
    def __init__(
        self,
        fisher_estimator: FisherEstimator = None,
        collector: GradientCollector = None,
        max_directions: int = 2000
    ):
        self.fisher_estimator = fisher_estimator or DiagonalFisherEstimator()
        self.collector = collector or AVECollector()
        # Change memory mode to 'orthonormal' to force normalization
        self.memory = GradientMemory(mode='raw', max_directions=max_directions)
        self.F_old: Optional[torch.Tensor] = None
        self.is_diagonal = isinstance(self.fisher_estimator, DiagonalFisherEstimator)
    
    def setup(self, model: nn.Module, config: Config):
        self.memory.clear()
        self.F_old = None
        self.lambda_reg = config.fopng_lambda_reg

    def _compute_update_prep(
        self,
        F_new: torch.Tensor,
        F_old: torch.Tensor,
        G: torch.Tensor,
        device: str
    ):
        """Precompute terms for FOPNG update if needed."""
        lam = self.lambda_reg

        if self.is_diagonal:
            # Diagonal Fisher approximation
            F_new_inv_diag = 1.0 / (F_new + lam)
            F_old_diag = F_old.view(-1, 1)
            F_old_G = F_old_diag * G
            weighted_G = F_old_diag * (F_new_inv_diag.view(-1, 1) * F_old_G)
            A = G.T @ weighted_G + lam * torch.eye(G.size(1), device=device)

            self.A_inv = torch.pinverse(A)
            # Also store A for condition number check
            self.A = A
        else:
            raise NotImplementedError("Precomputation for full Fisher not implemented.")

    
    def _compute_update(
        self,
        gradient: torch.Tensor,
        F_new: torch.Tensor,
        F_old: torch.Tensor,
        G: torch.Tensor,
        device: str,
        lr: float,
        return_intermediate: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Compute FOPNG update step.
        
        If return_intermediate=True, returns (update, stats_dict) where stats_dict contains:
        - raw_grad_norm: ||g||
        - fisher_grad_norm: ||F_old^{1/2} * g|| (gradient in Fisher space)
        - correction_norm: ||correction||
        - projected_grad_eucl_norm: ||P_g|| (projected gradient in Euclidean space)
        - projected_grad_fisher_norm: ||F_old^{1/2} * P_g|| (projected gradient in Fisher space)
        - update_norm: ||v_star|| (final update)
        - projection_relative_change: ||g - P_g|| / ||g||
        - fisher_projection_relative_change: ||F_old^{1/2} * (g - P_g)|| / ||F_old^{1/2} * g||
        - correction_to_raw_ratio: ||correction|| / ||g||
        - update_to_raw_ratio: ||v_star|| / ||g||
        """
        lam = self.lambda_reg
        stats = {}

        if self.is_diagonal:
            # Transform to Fisher space: g_F = F_old^{1/2} * g
            F_old_sqrt = torch.sqrt(F_old + 1e-10)
            g_fisher = F_old_sqrt * gradient
            
            F_new_inv_diag = 1.0 / (F_new + lam)
            
            # Original projection logic
            F_old_g = F_old * gradient
            G_T_F_old_g = G.T @ F_old_g
            A_inv_G_T_F_old_g = self.A_inv @ G_T_F_old_g
            correction = (G @ A_inv_G_T_F_old_g).view(-1) * F_old.squeeze()
            P_g = gradient - correction

            if return_intermediate:
                # --- DIAGNOSTICS FOR CORRECTION ---
                # Calculate norms for pipeline steps
                step1_norm = F_old_g.norm().item()
                step2_norm = G_T_F_old_g.norm().item()
                step3_norm = A_inv_G_T_F_old_g.norm().item()
                correction_unweighted = (G @ A_inv_G_T_F_old_g).view(-1)
                step4_norm = correction_unweighted.norm().item()
                step5_norm = correction.norm().item()
                grad_norm = gradient.norm().item()
                
                # Calculate matrix properties
                g_norm = G.norm().item()
                dim = G.shape[0]
                g_normalized = g_norm / (np.sqrt(dim) + 1e-10)
                
                # Calculate A matrix conditioning
                a_norm = torch.norm(self.A).item()
                a_inv_norm = torch.norm(self.A_inv).item()
                if device != 'mps':
                    a_cond = torch.linalg.cond(self.A).item()
                else:
                    # MPS backend does not support cond()
                    a_cond = float('0.0')

                # Ratios
                ratio1 = step1_norm / (grad_norm + 1e-10)
                ratio2 = step2_norm / (step1_norm + 1e-10)
                ratio3 = step3_norm / (step2_norm + 1e-10)
                ratio4 = step4_norm / (step3_norm + 1e-10)
                ratio5 = step5_norm / (step4_norm + 1e-10)

                stats.update({
                    'correction/step1_norm_F_old_g': step1_norm,
                    'correction/step2_norm_G_T_F_old_g': step2_norm,
                    'correction/step3_norm_A_inv_G_T': step3_norm,
                    'correction/step4_norm_correction_unweighted': step4_norm,
                    'correction/step5_norm_correction_final': step5_norm,
                    
                    'correction/ratio1_Fold_g_vs_g': ratio1,
                    'correction/ratio2_proj_vs_Fold_g': ratio2,
                    'correction/ratio3_Ainv_vs_proj': ratio3,
                    'correction/ratio4_unweighted_vs_Ainv': ratio4,
                    'correction/ratio5_final_vs_unweighted': ratio5,
                    
                    'correction/G_norm_normalized': g_normalized,
                    'correction/G_abs_mean': G.abs().mean().item(),
                    'correction/A_condition_number': a_cond,
                    'correction/A_norm': a_norm,
                    'correction/A_inv_norm': a_inv_norm,
                    
                    'correction/lambda_ratio_Fold': F_old.norm().item() / (lam + 1e-10),
                })
            
            # Projected gradient in Fisher space: F_old^{1/2} * P_g
            P_g_fisher = F_old_sqrt * P_g
            
            F_new_inv_P_g = P_g * F_new_inv_diag
            denom = torch.sqrt((P_g * F_new_inv_P_g).sum() + 1e-8)
            v_star = -lr * F_new_inv_P_g / (denom + 1e-8)
            
            if return_intermediate:
                # Compute comprehensive statistics
                raw_norm = gradient.norm().item()
                fisher_norm = g_fisher.norm().item()
                correction_norm = correction.norm().item()
                P_g_eucl_norm = P_g.norm().item()
                P_g_fisher_norm = P_g_fisher.norm().item()
                v_star_norm = v_star.norm().item()
                
                # Relative changes
                diff_eucl = (gradient - P_g).norm().item()
                diff_fisher = (g_fisher - P_g_fisher).norm().item()
                projection_relative_change = diff_eucl / (raw_norm + 1e-10)
                fisher_projection_relative_change = diff_fisher / (fisher_norm + 1e-10)
                
                stats.update({
                    'raw_grad_norm': raw_norm,
                    'fisher_grad_norm': fisher_norm,
                    'correction_norm': correction_norm,
                    'projected_grad_eucl_norm': P_g_eucl_norm,
                    'projected_grad_fisher_norm': P_g_fisher_norm,
                    'update_norm': v_star_norm,
                    'projection_relative_change': projection_relative_change,
                    'fisher_projection_relative_change': fisher_projection_relative_change,
                    'correction_to_raw_ratio': correction_norm / (raw_norm + 1e-10),
                    'update_to_raw_ratio': v_star_norm / (raw_norm + 1e-10),
                    'projected_to_raw_ratio_eucl': P_g_eucl_norm / (raw_norm + 1e-10),
                    'projected_to_raw_ratio_fisher': P_g_fisher_norm / (fisher_norm + 1e-10),
                })
        else:
            # Full Fisher
            # Transform to Fisher space
            F_old_sqrt = torch.linalg.cholesky(F_old + lam * torch.eye(F_old.size(0), device=device))
            g_fisher = F_old_sqrt @ gradient
            
            F_new_inv = torch.inverse(F_new + lam * torch.eye(F_new.size(0), device=device))
            
            temp = F_old @ F_new_inv @ F_old @ G
            A = G.T @ temp + lam * torch.eye(G.size(1), device=device)
            A_inv = torch.inverse(A)
            P = torch.eye(gradient.size(0), device=device) - F_old @ G @ A_inv @ G.T @ F_old
            P_g = P @ gradient
            
            # Projected gradient in Fisher space
            P_g_fisher = F_old_sqrt @ P_g
            
            F_new_inv_P_g = F_new_inv @ P_g
            denom = torch.sqrt(P_g @ F_new_inv_P_g + 1e-8)
            v_star = -lr * F_new_inv_P_g / denom
            
            if return_intermediate:
                # Compute comprehensive statistics
                raw_norm = gradient.norm().item()
                fisher_norm = g_fisher.norm().item()
                correction = gradient - P_g
                correction_norm = correction.norm().item()
                P_g_eucl_norm = P_g.norm().item()
                P_g_fisher_norm = P_g_fisher.norm().item()
                v_star_norm = v_star.norm().item()
                
                # Relative changes
                diff_eucl = correction_norm
                diff_fisher = (g_fisher - P_g_fisher).norm().item()
                projection_relative_change = diff_eucl / (raw_norm + 1e-10)
                fisher_projection_relative_change = diff_fisher / (fisher_norm + 1e-10)
                
                stats.update({
                    'raw_grad_norm': raw_norm,
                    'fisher_grad_norm': fisher_norm,
                    'correction_norm': correction_norm,
                    'projected_grad_eucl_norm': P_g_eucl_norm,
                    'projected_grad_fisher_norm': P_g_fisher_norm,
                    'update_norm': v_star_norm,
                    'projection_relative_change': projection_relative_change,
                    'fisher_projection_relative_change': fisher_projection_relative_change,
                    'correction_to_raw_ratio': correction_norm / (raw_norm + 1e-10),
                    'update_to_raw_ratio': v_star_norm / (raw_norm + 1e-10),
                    'projected_to_raw_ratio_eucl': P_g_eucl_norm / (raw_norm + 1e-10),
                    'projected_to_raw_ratio_fisher': P_g_fisher_norm / (fisher_norm + 1e-10),
                })
        
        if return_intermediate:
            return v_star, stats
        return v_star
    
    def train_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        criterion: nn.Module,
        config: Config,
        task_id: int,
        multihead: bool = False,
        progress_desc: Optional[str] = None
    ) -> Tuple[float, float]:
        
        # For first task or if no stored gradients, use regular training
        G = self.memory.get_matrix()
        if task_id == 0 or G is None:
            return self._train_regular(
                model, optimizer, train_loader, criterion, config,
                task_id, multihead, progress_desc
            )
        
        # Compute Fisher matrices
        F_new = self.fisher_estimator.estimate(model, train_loader, criterion, config.device)
        
        if self.F_old is None:
            self.F_old = F_new.clone()
        
        # Log Fisher matrix properties (computed once per epoch)
        fisher_new_norm = F_new.norm().item()
        fisher_old_norm = self.F_old.norm().item()
        fisher_diff_norm = (F_new - self.F_old).norm().item()
        fisher_relative_diff = fisher_diff_norm / (fisher_old_norm + 1e-10)
        
        # For diagonal Fisher, also track trace (sum of diagonal)
        if self.is_diagonal:
            fisher_new_trace = F_new.sum().item()
            fisher_old_trace = self.F_old.sum().item()
            fisher_trace_diff = abs(fisher_new_trace - fisher_old_trace)
            fisher_trace_relative_diff = fisher_trace_diff / (abs(fisher_old_trace) + 1e-10)
        else:
            fisher_new_trace = torch.trace(F_new).item() if F_new.dim() == 2 else None
            fisher_old_trace = torch.trace(self.F_old).item() if self.F_old.dim() == 2 else None
            fisher_trace_diff = abs(fisher_new_trace - fisher_old_trace) if fisher_new_trace is not None else None
            fisher_trace_relative_diff = fisher_trace_diff / (abs(fisher_old_trace) + 1e-10) if fisher_old_trace is not None else None
        
        # Log number of stored gradient directions
        num_directions = G.size(1) if G is not None else 0
        
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        self._compute_update_prep(F_new, self.F_old, G, config.device)
        iterator = tqdm(train_loader, desc=progress_desc, leave=False) if progress_desc else train_loader
        
        # Accumulators for all metrics (log average per epoch)
        batch_stats = {
            'raw_grad_norm': [],
            'fisher_grad_norm': [],
            'correction_norm': [],
            'projected_grad_eucl_norm': [],
            'projected_grad_fisher_norm': [],
            'update_norm': [],
            'projection_relative_change': [],
            'fisher_projection_relative_change': [],
            'correction_to_raw_ratio': [],
            'update_to_raw_ratio': [],
            'projected_to_raw_ratio_eucl': [],
            'projected_to_raw_ratio_fisher': [],
            
            # New diagnostics
            'correction/step1_norm_F_old_g': [],
            'correction/step2_norm_G_T_F_old_g': [],
            'correction/step3_norm_A_inv_G_T': [],
            'correction/step4_norm_correction_unweighted': [],
            'correction/step5_norm_correction_final': [],
            
            'correction/ratio1_Fold_g_vs_g': [],
            'correction/ratio2_proj_vs_Fold_g': [],
            'correction/ratio3_Ainv_vs_proj': [],
            'correction/ratio4_unweighted_vs_Ainv': [],
            'correction/ratio5_final_vs_unweighted': [],
            
            'correction/G_norm_normalized': [],
            'correction/G_abs_mean': [],
            'correction/A_condition_number': [],
            'correction/A_norm': [],
            'correction/A_inv_norm': [],
            'correction/lambda_ratio_Fold': [],
        }
        
        for x, y in iterator:
            x = x.to(config.device)
            y = y.to(config.device)
            
            if multihead:
                output = model(x, task_id=task_id)
            else:
                output = model(x)
            
            loss = criterion(output, y)
            model.zero_grad()
            loss.backward()
            
            grad = get_grad_vector(model)
            update, stats = self._compute_update(grad, F_new, self.F_old, G, config.device, config.lr, return_intermediate=True)
            apply_update(model, update)
            
            # Accumulate all statistics
            for key in batch_stats.keys():
                if key in stats:
                    batch_stats[key].append(stats[key])
            
            total_loss += loss.item() * x.size(0)
            preds = output.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
        
        # Compute means for all metrics
        log_metrics = {
            "task_id": task_id,
        }
        
        # Fisher matrix metrics (per epoch)
        log_metrics.update({
            f"fisher_matrix/fisher_new_norm": fisher_new_norm,
            f"fisher_matrix/fisher_old_norm": fisher_old_norm,
            f"fisher_matrix/fisher_diff_norm": fisher_diff_norm,
            f"fisher_matrix/fisher_relative_diff": fisher_relative_diff,
            f"fisher_matrix/num_directions": num_directions,
        })
        
        if fisher_new_trace is not None:
            log_metrics.update({
                f"fisher_matrix/fisher_new_trace": fisher_new_trace,
                f"fisher_matrix/fisher_old_trace": fisher_old_trace,
                f"fisher_matrix/fisher_trace_diff": fisher_trace_diff,
                f"fisher_matrix/fisher_trace_relative_diff": fisher_trace_relative_diff,
            })
        
        # Gradient and update metrics (averaged over batches in epoch)
        for key, values in batch_stats.items():
            if values:
                if key.startswith('correction/'):
                    # Log correction metrics directly under correction/
                    log_metrics[f"{key}_mean"] = np.mean(values)
                    log_metrics[f"{key}_std"] = np.std(values)
                else:
                    # Log other gradients under fopng_gradients/
                    log_metrics[f"fopng_gradients/{key}_mean"] = np.mean(values)
                    log_metrics[f"fopng_gradients/{key}_std"] = np.std(values)
        
        # Log all metrics to wandb
        log(log_metrics)
        
        return total_loss / total_samples, total_correct / total_samples
    
    def _train_regular(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        criterion: nn.Module,
        config: Config,
        task_id: int,
        multihead: bool = False,
        progress_desc: Optional[str] = None
    ) -> Tuple[float, float]:
        """Regular Adam training for first task."""
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # Accumulators for gradient norms (log average per epoch)
        raw_grad_norms = []
        
        iterator = tqdm(train_loader, desc=progress_desc, leave=False) if progress_desc else train_loader
        
        for x, y in iterator:
            x = x.to(config.device)
            y = y.to(config.device)
            
            optimizer.zero_grad()
            
            if multihead:
                logits = model(x, task_id=task_id)
            else:
                logits = model(x)
            
            loss = criterion(logits, y)
            loss.backward()
            
            # Get raw gradient norm before optimizer step
            grad = get_grad_vector(model)
            raw_grad_norms.append(grad.norm().item())
            
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
        
        # Log average gradient norms for this epoch (task 0, no FOPNG projection)
        log({
            f"fopng_gradients/raw_grad_norm_mean": np.mean(raw_grad_norms),
            f"fopng_gradients/raw_grad_norm_std": np.std(raw_grad_norms),
            "task_id": task_id,
        })
        
        return total_loss / total_samples, total_correct / total_samples
    
    def after_task(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        task_id: int,
        config: Config,
        multihead: bool = False
    ):
        # Update F_old with current task's Fisher
        criterion = nn.CrossEntropyLoss()
        F_current = self.fisher_estimator.estimate(model, train_loader, criterion, config.device)
        
        if self.F_old is None:
            self.F_old = F_current
        else:
            w = getattr(config, 'fopng_new_fisher_weight')
            self.F_old = (1 - w) * self.F_old + w * F_current
        
        # Collect gradients
        print(f"Collecting FOPNG directions from task {task_id}...")
        self.collector.collect(
            self.memory,
            model,
            train_loader,
            config.grads_per_task,
            config.device,
            multihead=multihead,
            task_id=task_id if multihead else None
        )

class FNGMethod(ContinualMethod):
    """
    Fisher Natural Gradient.
    Uses Fisher information to define a Riemannian metric for projection.
    """
    
    def __init__(
        self,
        fisher_estimator: FisherEstimator = None,
    ):
        self.fisher_estimator = fisher_estimator or DiagonalFisherEstimator()
        self.is_diagonal = isinstance(self.fisher_estimator, DiagonalFisherEstimator)
    
    def setup(self, model: nn.Module, config: Config):
        self.lambda_reg = config.fopng_lambda_reg
    
    def _compute_update(
        self,
        gradient: torch.Tensor,
        F_new: torch.Tensor,
        device: str,
        lr: float,
        return_intermediate: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Compute FOPNG update step.
        
        If return_intermediate=True, returns (update, stats_dict) where stats_dict contains:
        - raw_grad_norm: ||g||
        - update_norm: ||v_star|| (final update)
        - update_to_raw_ratio: ||v_star|| / ||g||
        """
        lam = self.lambda_reg
        stats = {}

        if self.is_diagonal:
            F_inv_diag = 1.0 / (F_new + lam)
            denom = torch.sqrt((gradient * (F_inv_diag * gradient)).sum())
            v_star = -lr * F_inv_diag * gradient / (denom + 1e-8)
            
            if return_intermediate:
                # Compute comprehensive statistics
                raw_norm = gradient.norm().item()
                v_star_norm = v_star.norm().item()
                
                stats.update({
                    'raw_grad_norm': raw_norm,
                    'update_norm': v_star_norm,
                    'update_to_raw_ratio': v_star_norm / (raw_norm + 1e-10),
                })
        else:
            raise NotImplemented("FNG only supports diagonal Fisher estimator.")
        
        if return_intermediate:
            return v_star, stats
        return v_star
    
    def train_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        criterion: nn.Module,
        config: Config,
        task_id: int,
        multihead: bool = False,
        progress_desc: Optional[str] = None
    ) -> Tuple[float, float]:
        
        # For first task or if no stored gradients, use regular training
        if task_id == 0:
            return self._train_regular(
                model, optimizer, train_loader, criterion, config,
                task_id, multihead, progress_desc
            )
        
        # Compute Fisher matrices
        F_new = self.fisher_estimator.estimate(model, train_loader, criterion, config.device)
        
        # Log Fisher matrix properties (computed once per epoch)
        fisher_new_norm = F_new.norm().item()
        
        # For diagonal Fisher, also track trace (sum of diagonal)
        if self.is_diagonal:
            fisher_new_trace = F_new.sum().item()
        else:
            fisher_new_trace = 0.0
            raise NotImplemented("FNG only supports diagonal Fisher estimator.")
        
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        iterator = tqdm(train_loader, desc=progress_desc, leave=False) if progress_desc else train_loader
        
        # Accumulators for all metrics (log average per epoch)
        batch_stats = {
            'raw_grad_norm': [],
            'update_norm': [],
            'update_to_raw_ratio': [],  
        }
        
        for x, y in iterator:
            x = x.to(config.device)
            y = y.to(config.device)
            
            if multihead:
                output = model(x, task_id=task_id)
            else:
                output = model(x)
            
            loss = criterion(output, y)
            model.zero_grad()
            loss.backward()
            
            grad = get_grad_vector(model)
            update, stats = self._compute_update(grad, F_new, config.device, config.lr, return_intermediate=True)
            apply_update(model, update)
            
            # Accumulate all statistics
            for key in batch_stats.keys():
                if key in stats:
                    batch_stats[key].append(stats[key])
            
            total_loss += loss.item() * x.size(0)
            preds = output.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
        
        # Compute means for all metrics
        log_metrics = {
            "task_id": task_id,
        }
        
        # Fisher matrix metrics (per epoch)
        log_metrics.update({
            f"fisher_matrix/fisher_new_norm": fisher_new_norm,
        })
        
        if fisher_new_trace is not None:
            log_metrics.update({
                f"fisher_matrix/fisher_new_trace": fisher_new_trace,
            })
        
        # Gradient and update metrics (averaged over batches in epoch)
        for key, values in batch_stats.items():
            if values:
                if key.startswith('correction/'):
                    # Log correction metrics directly under correction/
                    log_metrics[f"{key}_mean"] = np.mean(values)
                    log_metrics[f"{key}_std"] = np.std(values)
                else:
                    # Log other gradients under fopng_gradients/
                    log_metrics[f"fopng_gradients/{key}_mean"] = np.mean(values)
                    log_metrics[f"fopng_gradients/{key}_std"] = np.std(values)
        
        # Log all metrics to wandb
        log(log_metrics)
        
        return total_loss / total_samples, total_correct / total_samples
    
    def _train_regular(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        criterion: nn.Module,
        config: Config,
        task_id: int,
        multihead: bool = False,
        progress_desc: Optional[str] = None
    ) -> Tuple[float, float]:
        """Regular Adam training for first task."""
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # Accumulators for gradient norms (log average per epoch)
        raw_grad_norms = []
        
        iterator = tqdm(train_loader, desc=progress_desc, leave=False) if progress_desc else train_loader
        
        for x, y in iterator:
            x = x.to(config.device)
            y = y.to(config.device)
            
            optimizer.zero_grad()
            
            if multihead:
                logits = model(x, task_id=task_id)
            else:
                logits = model(x)
            
            loss = criterion(logits, y)
            loss.backward()
            
            # Get raw gradient norm before optimizer step
            grad = get_grad_vector(model)
            raw_grad_norms.append(grad.norm().item())
            
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
        
        # Log average gradient norms for this epoch (task 0, no FOPNG projection)
        log({
            f"fopng_gradients/raw_grad_norm_mean": np.mean(raw_grad_norms),
            f"fopng_gradients/raw_grad_norm_std": np.std(raw_grad_norms),
            "task_id": task_id,
        })
        
        return total_loss / total_samples, total_correct / total_samples
    
    def after_task(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        task_id: int,
        config: Config,
        multihead: bool = False
    ):
        pass


class EWCMethod(ContinualMethod):
    """
    Elastic Weight Consolidation (EWC).
    
    Adds a quadratic penalty to the loss to preserve important parameters
    from previous tasks: L_EWC = L_task + (λ/2) Σ F_i (θ_i - θ*_i)^2
    
    where F_i is the diagonal Fisher information and θ*_i are the optimal
    parameters from previous tasks.
    """
    
    def __init__(self, fisher_estimator: FisherEstimator = None):
        self.fisher_estimator = fisher_estimator or DiagonalFisherEstimator()
        self.is_diagonal = isinstance(self.fisher_estimator, DiagonalFisherEstimator)
        
        # Store consolidated Fisher information and optimal parameters from all previous tasks
        self.fisher_dict: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        self.ewc_lambda: float = 0.0
    
    def setup(self, model: nn.Module, config: Config):
        """Initialize EWC state."""
        self.fisher_dict.clear()
        self.optimal_params.clear()
        self.ewc_lambda = config.ewc_lambda
    
    def _compute_ewc_loss(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC regularization loss: (λ/2) Σ F_i (θ_i - θ*_i)^2
        """
        if not self.fisher_dict:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        ewc_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for name, param in model.named_parameters():
            if name in self.fisher_dict:
                fisher = self.fisher_dict[name]
                optimal = self.optimal_params[name]
                # Quadratic penalty weighted by Fisher information
                ewc_loss += (fisher * (param - optimal).pow(2)).sum()
        
        return (self.ewc_lambda / 2.0) * ewc_loss
    
    def train_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        criterion: nn.Module,
        config: Config,
        task_id: int,
        multihead: bool = False,
        progress_desc: Optional[str] = None
    ) -> Tuple[float, float]:
        model.train()
        total_loss = 0.0
        total_task_loss = 0.0
        total_ewc_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        iterator = tqdm(train_loader, desc=progress_desc, leave=False) if progress_desc else train_loader
        
        for x, y in iterator:
            x = x.to(config.device)
            y = y.to(config.device)
            
            optimizer.zero_grad()
            
            if multihead:
                logits = model(x, task_id=task_id)
            else:
                logits = model(x)
            
            # Task loss
            task_loss = criterion(logits, y)
            
            # EWC penalty (only applied after first task)
            ewc_loss = self._compute_ewc_loss(model) if task_id > 0 else torch.tensor(0.0, device=config.device)
            
            # Total loss
            loss = task_loss + ewc_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            total_task_loss += task_loss.item() * x.size(0)
            total_ewc_loss += ewc_loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
        
        avg_loss = total_loss / total_samples
        avg_task_loss = total_task_loss / total_samples
        avg_ewc_loss = total_ewc_loss / total_samples
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def after_task(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        task_id: int,
        config: Config,
        multihead: bool = False
    ):
        """
        After completing a task:
        1. Compute Fisher information on current task
        2. Store current parameters as optimal
        3. Accumulate Fisher (sum across tasks)
        """
        print(f"Computing Fisher information for EWC after task {task_id}...")
        
        # Estimate Fisher on current task
        criterion = nn.CrossEntropyLoss()
        
        # Create subset of data if ewc_fisher_samples is specified
        if config.ewc_fisher_samples is not None and config.ewc_fisher_samples > 0:
            from torch.utils.data import Subset
            import random
            indices = random.sample(range(len(train_loader.dataset)), 
                                  min(config.ewc_fisher_samples, len(train_loader.dataset)))
            subset = Subset(train_loader.dataset, indices)
            fisher_loader = DataLoader(subset, batch_size=train_loader.batch_size, shuffle=False)
        else:
            fisher_loader = train_loader
        
        fisher_flat = self.fisher_estimator.estimate(model, fisher_loader, criterion, config.device)
        
        # Convert flat Fisher back to dict
        idx = 0
        task_fisher = {}
        for name, param in model.named_parameters():
            num_params = param.numel()
            task_fisher[name] = fisher_flat[idx:idx+num_params].view_as(param).clone()
            idx += num_params
        
        # Accumulate Fisher information (sum over tasks - original EWC)
        for name, param in model.named_parameters():
            if name in self.fisher_dict:
                self.fisher_dict[name] += task_fisher[name]
            else:
                self.fisher_dict[name] = task_fisher[name].clone()
            
            # Store optimal parameters
            self.optimal_params[name] = param.data.clone()
        
        print(f"EWC: Fisher information computed and accumulated for task {task_id}")