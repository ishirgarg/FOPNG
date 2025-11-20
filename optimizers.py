import torch
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Dict, Any, List

from config import Config
from gradients import GradientMemory, GradientCollector, GTLCollector, AVECollector
from fisher import FisherEstimator, DiagonalFisherEstimator, FullFisherEstimator
from utils import get_param_count, apply_update
from gradients import get_grad_vector, set_grad_vector

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
        collect_stats: bool = False
    ) -> Tuple[float, float, Optional[Dict[str, Any]]]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (loss, accuracy, optional_stats_dict)
            Stats dict may contain: grad_norm_mean, grad_norm_std, update_norm_mean, update_norm_std
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
        collect_stats: bool = False
    ) -> Tuple[float, float, Optional[Dict[str, Any]]]:
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        grad_norms = []
        update_norms = []
        
        for x, y in train_loader:
            x = x.to(config.device)
            y = y.to(config.device)
            
            optimizer.zero_grad()
            
            if multihead:
                logits = model(x, task_id=task_id)
            else:
                logits = model(x)
            
            loss = criterion(logits, y)
            loss.backward()
            
            if collect_stats:
                grad = get_grad_vector(model)
                grad_norms.append(grad.norm().item())
                
                old_params = torch.cat([p.data.view(-1).clone() for p in model.parameters()])
            
            optimizer.step()
            
            if collect_stats:
                new_params = torch.cat([p.data.view(-1) for p in model.parameters()])
                update_norms.append((new_params - old_params).norm().item())
            
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
        
        stats = None
        if collect_stats and grad_norms:
            stats = {
                'grad_norm_mean': np.mean(grad_norms),
                'grad_norm_std': np.std(grad_norms),
                'update_norm_mean': np.mean(update_norms),
                'update_norm_std': np.std(update_norms),
            }
        
        return total_loss / total_samples, total_correct / total_samples, stats
    
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
        collect_stats: bool = False
    ) -> Tuple[float, float, Optional[Dict[str, Any]]]:
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        grad_norms = []
        projected_grad_norms = []
        update_norms = []
        
        for x, y in train_loader:
            x = x.to(config.device)
            y = y.to(config.device)
            
            optimizer.zero_grad()
            
            if multihead:
                logits = model(x, task_id=task_id)
            else:
                logits = model(x)
            
            loss = criterion(logits, y)
            loss.backward()
            
            # Project gradient if we have stored directions
            if len(self.memory) > 0:
                g = get_grad_vector(model)
                if collect_stats:
                    grad_norms.append(g.norm().item())
                g_tilde = self.memory.project_orthogonal(g)
                if collect_stats:
                    projected_grad_norms.append(g_tilde.norm().item())
                set_grad_vector(model, g_tilde)
            elif collect_stats:
                g = get_grad_vector(model)
                grad_norms.append(g.norm().item())
            
            if collect_stats:
                old_params = torch.cat([p.data.view(-1).clone() for p in model.parameters()])
            
            optimizer.step()
            
            if collect_stats:
                new_params = torch.cat([p.data.view(-1) for p in model.parameters()])
                update_norms.append((new_params - old_params).norm().item())
            
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
        
        stats = None
        if collect_stats and grad_norms:
            stats = {
                'grad_norm_mean': np.mean(grad_norms),
                'grad_norm_std': np.std(grad_norms),
                'update_norm_mean': np.mean(update_norms),
                'update_norm_std': np.std(update_norms),
            }
            if projected_grad_norms:
                stats['projected_grad_norm_mean'] = np.mean(projected_grad_norms)
                stats['projected_grad_norm_std'] = np.std(projected_grad_norms)
        
        return total_loss / total_samples, total_correct / total_samples, stats
    
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
        max_directions: int = 2000,
        use_cumulative_fisher: bool = False
    ):
        self.fisher_estimator = fisher_estimator or DiagonalFisherEstimator()
        self.collector = collector or AVECollector()
        self.memory = GradientMemory(mode='raw', max_directions=max_directions)
        self.F_old: Optional[torch.Tensor] = None
        self.is_diagonal = isinstance(self.fisher_estimator, DiagonalFisherEstimator)
        self.use_cumulative_fisher = use_cumulative_fisher
        self.num_tasks_seen = 0
    
    def setup(self, model: nn.Module, config: Config):
        self.memory.clear()
        self.F_old = None
        self.lambda_reg = config.fopng_lambda_reg
        self.epsilon = config.fopng_epsilon
        self.num_tasks_seen = 0

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
        else:
            raise NotImplementedError("Precomputation for full Fisher not implemented.")

    
    def _compute_update(
        self,
        gradient: torch.Tensor,
        F_new: torch.Tensor,
        F_old: torch.Tensor,
        G: torch.Tensor,
        device: str
    ) -> torch.Tensor:
        """Compute FOPNG update step."""
        lam = self.lambda_reg

        if self.is_diagonal:
            F_new_inv_diag = 1.0 / (F_new + lam)
            F_old_g = F_old * gradient
            G_T_F_old_g = G.T @ F_old_g
            A_inv_G_T_F_old_g = self.A_inv @ G_T_F_old_g
            correction = (G @ A_inv_G_T_F_old_g).view(-1) * F_old.squeeze()
            P_g = gradient - correction
            F_new_inv_P_g = P_g * F_new_inv_diag
            denom = torch.sqrt((P_g * F_new_inv_P_g).sum() + 1e-8)
            v_star = -self.epsilon * F_new_inv_P_g / (denom + 1e-8)
        else:
            # Full Fisher
            F_new_inv = torch.inverse(F_new + lam * torch.eye(F_new.size(0), device=device))
            temp = F_old @ F_new_inv @ F_old @ G
            A = G.T @ temp + lam * torch.eye(G.size(1), device=device)
            A_inv = torch.inverse(A)
            P = torch.eye(gradient.size(0), device=device) - F_old @ G @ A_inv @ G.T @ F_old
            P_g = P @ gradient
            F_new_inv_P_g = F_new_inv @ P_g
            denom = torch.sqrt(P_g @ F_new_inv_P_g + 1e-8)
            v_star = -self.epsilon * F_new_inv_P_g / denom
        
        return v_star
    
    def _compute_update_detailed(
        self,
        gradient: torch.Tensor,
        F_new: torch.Tensor,
        F_old: torch.Tensor,
        G: torch.Tensor,
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute FOPNG update step with detailed debug info."""
        lam = self.lambda_reg
        debug_info = {}

        if self.is_diagonal:
            F_new_inv_diag = 1.0 / (F_new + lam)
            F_old_g = F_old * gradient
            G_T_F_old_g = G.T @ F_old_g
            A_inv_G_T_F_old_g = self.A_inv @ G_T_F_old_g
            correction = (G @ A_inv_G_T_F_old_g).view(-1) * F_old.squeeze()
            P_g = gradient - correction
            F_new_inv_P_g = P_g * F_new_inv_diag
            denom = torch.sqrt((P_g * F_new_inv_P_g).sum() + 1e-8)
            v_star = -self.epsilon * F_new_inv_P_g / (denom + 1e-8)
            
            # Debug info
            debug_info['correction'] = correction.detach()
            debug_info['correction_norm'] = correction.norm().item()
            debug_info['projected_grad_norm'] = P_g.norm().item()
            debug_info['projection_ratio'] = correction.norm().item() / (gradient.norm().item() + 1e-8)
            debug_info['fisher_norm'] = denom.item()
            debug_info['denom'] = denom.item()
        else:
            # Full Fisher
            F_new_inv = torch.inverse(F_new + lam * torch.eye(F_new.size(0), device=device))
            temp = F_old @ F_new_inv @ F_old @ G
            A = G.T @ temp + lam * torch.eye(G.size(1), device=device)
            A_inv = torch.inverse(A)
            P = torch.eye(gradient.size(0), device=device) - F_old @ G @ A_inv @ G.T @ F_old
            P_g = P @ gradient
            F_new_inv_P_g = F_new_inv @ P_g
            denom = torch.sqrt(P_g @ F_new_inv_P_g + 1e-8)
            v_star = -self.epsilon * F_new_inv_P_g / denom
            
            # Debug info
            correction = gradient - P_g
            debug_info['correction'] = correction.detach()
            debug_info['correction_norm'] = correction.norm().item()
            debug_info['projected_grad_norm'] = P_g.norm().item()
            debug_info['projection_ratio'] = correction.norm().item() / (gradient.norm().item() + 1e-8)
            debug_info['fisher_norm'] = denom.item()
            debug_info['denom'] = denom.item()
        
        return v_star, debug_info
    
    def train_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        criterion: nn.Module,
        config: Config,
        task_id: int,
        multihead: bool = False,
        collect_stats: bool = False
    ) -> Tuple[float, float, Optional[Dict[str, Any]]]:
        
        # For first task or if no stored gradients, use regular training
        G = self.memory.get_matrix()
        if task_id == 0 or G is None:
            return self._train_regular(model, optimizer, train_loader, criterion, config, task_id, multihead, collect_stats)
        
        # Compute Fisher matrices
        
        F_new = self.fisher_estimator.estimate(model, train_loader, criterion, config.device)
        
        if self.F_old is None:
            self.F_old = F_new.clone()

        # EXTRA CODE
        # self.F_old = F_new.clone()
        ##
        
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        grad_norms = []
        update_norms = []
        projection_ratios = []
        correction_norms = []
        fisher_norms = []
        
        self._compute_update_prep(F_new, self.F_old, G, config.device)
        
        batch_idx = 0
        for x, y in train_loader:
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
            grad_norm = grad.norm().item()
            
            # Compute update with detailed tracking
            update, debug_info = self._compute_update_detailed(grad, F_new, self.F_old, G, config.device)
            update_norm = update.norm().item()
            
            if collect_stats:
                grad_norms.append(grad_norm)
                update_norms.append(update_norm)
                projection_ratios.append(debug_info['projection_ratio'])
                correction_norms.append(debug_info['correction_norm'])
                fisher_norms.append(debug_info['fisher_norm'])
            
            # DEBUG: Print first batch of each epoch for task_id > 0
            if batch_idx == 0 and task_id > 0:
                print(f"  [FIRST BATCH DEBUG - Task {task_id}]")
                print(f"    Gradient L2 norm: {grad_norm:.6f}")
                print(f"    Correction L2 norm: {debug_info['correction_norm']:.6f}")
                print(f"    Projected grad L2 norm: {debug_info['projected_grad_norm']:.6f}")
                print(f"    Projection ratio (correction/gradient): {debug_info['projection_ratio']:.4f}")
                print(f"    Update L2 norm: {update_norm:.6f}")
                print(f"    Update Fisher norm: {debug_info['fisher_norm']:.6f}")
                print(f"    Gradient direction (first 10 elements): {grad[:10].cpu().numpy()}")
                print(f"    Correction direction (first 10 elements): {debug_info['correction'][:10].cpu().numpy()}")
                print(f"    Update direction (first 10 elements): {update[:10].cpu().numpy()}")
                print(f"    F_old stats: mean={self.F_old.mean().item():.6e}, norm={self.F_old.norm().item():.6f}")
                print(f"    F_new stats: mean={F_new.mean().item():.6e}, norm={F_new.norm().item():.6f}")
            
            apply_update(model, update)
            
            total_loss += loss.item() * x.size(0)
            preds = output.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
            batch_idx += 1
        
        stats = None
        if collect_stats and grad_norms:
            stats = {
                'grad_norm_mean': np.mean(grad_norms),
                'grad_norm_std': np.std(grad_norms),
                'update_norm_mean': np.mean(update_norms),
                'update_norm_std': np.std(update_norms),
                'projection_ratio_mean': np.mean(projection_ratios),
                'correction_norm_mean': np.mean(correction_norms),
                'fisher_norm_mean': np.mean(fisher_norms),
            }
        
        return total_loss / total_samples, total_correct / total_samples, stats
    
    def _train_regular(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        criterion: nn.Module,
        config: Config,
        task_id: int,
        multihead: bool = False,
        collect_stats: bool = False
    ) -> Tuple[float, float, Optional[Dict[str, Any]]]:
        """Regular Adam training for first task."""
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        grad_norms = []
        update_norms = []
        
        for x, y in train_loader:
            x = x.to(config.device)
            y = y.to(config.device)
            
            optimizer.zero_grad()
            
            if multihead:
                logits = model(x, task_id=task_id)
            else:
                logits = model(x)
            
            loss = criterion(logits, y)
            loss.backward()
            
            if collect_stats:
                grad = get_grad_vector(model)
                grad_norms.append(grad.norm().item())
                old_params = torch.cat([p.data.view(-1).clone() for p in model.parameters()])
            
            optimizer.step()
            
            if collect_stats:
                new_params = torch.cat([p.data.view(-1) for p in model.parameters()])
                update_norms.append((new_params - old_params).norm().item())
            
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
        
        stats = None
        if collect_stats and grad_norms:
            stats = {
                'grad_norm_mean': np.mean(grad_norms),
                'grad_norm_std': np.std(grad_norms),
                'update_norm_mean': np.mean(update_norms),
                'update_norm_std': np.std(update_norms),
            }
        
        return total_loss / total_samples, total_correct / total_samples, stats
    
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
        
        print(f"\n=== Fisher Matrix Update Debug (Task {task_id}) ===")
        print(f"F_current stats: mean={F_current.mean().item():.6f}, std={F_current.std().item():.6f}, max={F_current.max().item():.6f}")
        
        if self.F_old is None:
            self.F_old = F_current
            self.num_tasks_seen = 1
            print(f"First task - initializing F_old")
        else:
            print(f"F_old (before) stats: mean={self.F_old.mean().item():.6f}, std={self.F_old.std().item():.6f}, max={self.F_old.max().item():.6f}")
            
            # Calculate relative norm difference before update
            norm_diff = torch.norm(F_current - self.F_old).item()
            norm_f_old = torch.norm(self.F_old).item()
            relative_diff = norm_diff / norm_f_old if norm_f_old > 0 else 0.0
            print(f"||F_current - F_old|| / ||F_old|| = {relative_diff:.6f}")
            
            # Debug: Print norms before momentum update
            norm_f_new = torch.norm(F_current).item()
            print(f"DEBUG: ||F_new|| = {norm_f_new:.6f}")
            print(f"DEBUG: ||F_old|| = {norm_f_old:.6f}")
            
            if self.use_cumulative_fisher:
                # Cumulative average: equal weight to all tasks
                print(f"Using CUMULATIVE averaging (equal weight to all {self.num_tasks_seen + 1} tasks)")
                self.F_old = (self.num_tasks_seen * self.F_old + F_current) / (self.num_tasks_seen + 1)
                self.num_tasks_seen += 1
            else:
                # Exponential moving average (momentum)
                beta = config.fopng_fisher_momentum
                print(f"Using EMA with beta (momentum) = {beta:.3f}")
                print(f"  -> {beta:.3f} * F_old + {1-beta:.3f} * F_current")
                self.F_old = beta * self.F_old + (1 - beta) * F_current
            
            print(f"F_old (after) stats: mean={self.F_old.mean().item():.6f}, std={self.F_old.std().item():.6f}, max={self.F_old.max().item():.6f}")
        
        print(f"===========================================\n")
        
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