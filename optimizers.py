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
        collect_stats: bool = False,
        progress_desc: Optional[str] = None
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
        collect_stats: bool = False,
        progress_desc: Optional[str] = None
    ) -> Tuple[float, float, Optional[Dict[str, Any]]]:
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        grad_norms = []
        update_norms = []
        
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
        collect_stats: bool = False,
        progress_desc: Optional[str] = None
    ) -> Tuple[float, float, Optional[Dict[str, Any]]]:
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        grad_norms = []
        projected_grad_norms = []
        update_norms = []
        projection_relative_changes = []  # ||g - g_proj|| / ||g||
        
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
            
            # Project gradient if we have stored directions
            if len(self.memory) > 0:
                g = get_grad_vector(model)
                g_norm = g.norm().item()
                if collect_stats:
                    grad_norms.append(g_norm)
                g_tilde = self.memory.project_orthogonal(g)
                if collect_stats:
                    projected_grad_norms.append(g_tilde.norm().item())
                    # Compute relative change
                    diff_norm = (g - g_tilde).norm().item()
                    relative_change = diff_norm / (g_norm + 1e-10)
                    projection_relative_changes.append(relative_change)
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
            if projection_relative_changes:
                stats['ogd_projection_relative_change_mean'] = np.mean(projection_relative_changes)
                stats['ogd_projection_relative_change_std'] = np.std(projection_relative_changes)
        
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
        max_directions: int = 2000
    ):
        self.fisher_estimator = fisher_estimator or DiagonalFisherEstimator()
        self.collector = collector or AVECollector()
        self.memory = GradientMemory(mode='raw', max_directions=max_directions)
        self.F_old: Optional[torch.Tensor] = None
        self.is_diagonal = isinstance(self.fisher_estimator, DiagonalFisherEstimator)
    
    def setup(self, model: nn.Module, config: Config):
        self.memory.clear()
        self.F_old = None
        self.lambda_reg = config.fopng_lambda_reg
        self.epsilon = config.fopng_epsilon

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
        device: str,
        return_intermediate: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Compute FOPNG update step.
        
        If return_intermediate=True, returns (update, stats_dict) where stats_dict contains:
        - raw_grad_norm: ||g||
        - fisher_grad_norm: ||F_old^{1/2} * g||
        - fisher_proj_grad_norm: ||projected gradient in Fisher space||
        - eucl_proj_grad_norm: ||F_old^{-1/2} * projected gradient||
        - projected_grad: the projected gradient P_g (for computing relative change)
        """
        lam = self.lambda_reg
        
        intermediate_stats = {}
        if return_intermediate:
            intermediate_stats['raw_grad_norm'] = gradient.norm().item()

        if self.is_diagonal:
            # Transform to Fisher space: g_F = F_old^{1/2} * g
            F_old_sqrt = torch.sqrt(F_old + 1e-10)
            g_fisher = F_old_sqrt * gradient
            if return_intermediate:
                intermediate_stats['fisher_grad_norm'] = g_fisher.norm().item()
            
            F_new_inv_diag = 1.0 / (F_new + lam)
            F_old_g = F_old * gradient
            G_T_F_old_g = G.T @ F_old_g
            A_inv_G_T_F_old_g = self.A_inv @ G_T_F_old_g
            correction = (G @ A_inv_G_T_F_old_g).view(-1) * F_old.squeeze()
            P_g = gradient - correction
            
            # P_g is the projected gradient in Euclidean space
            # To get Fisher space projected gradient: F_old^{1/2} * P_g
            if return_intermediate:
                g_fisher_proj = F_old_sqrt * P_g
                intermediate_stats['fisher_proj_grad_norm'] = g_fisher_proj.norm().item()
                intermediate_stats['eucl_proj_grad_norm'] = P_g.norm().item()
                intermediate_stats['projected_grad'] = P_g
            
            F_new_inv_P_g = P_g * F_new_inv_diag
            denom = torch.sqrt((P_g * F_new_inv_P_g).sum() + 1e-8)
            v_star = -self.epsilon * F_new_inv_P_g / (denom + 1e-8)
        else:
            # Full Fisher
            # Transform to Fisher space
            F_old_sqrt = torch.linalg.cholesky(F_old + lam * torch.eye(F_old.size(0), device=device))
            g_fisher = F_old_sqrt @ gradient
            if return_intermediate:
                intermediate_stats['fisher_grad_norm'] = g_fisher.norm().item()
            
            F_new_inv = torch.inverse(F_new + lam * torch.eye(F_new.size(0), device=device))
            temp = F_old @ F_new_inv @ F_old @ G
            A = G.T @ temp + lam * torch.eye(G.size(1), device=device)
            A_inv = torch.inverse(A)
            P = torch.eye(gradient.size(0), device=device) - F_old @ G @ A_inv @ G.T @ F_old
            P_g = P @ gradient
            
            if return_intermediate:
                g_fisher_proj = F_old_sqrt @ P_g
                intermediate_stats['fisher_proj_grad_norm'] = g_fisher_proj.norm().item()
                intermediate_stats['eucl_proj_grad_norm'] = P_g.norm().item()
                intermediate_stats['projected_grad'] = P_g
            
            F_new_inv_P_g = F_new_inv @ P_g
            denom = torch.sqrt(P_g @ F_new_inv_P_g + 1e-8)
            v_star = -self.epsilon * F_new_inv_P_g / denom
        
        if return_intermediate:
            return v_star, intermediate_stats
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
        collect_stats: bool = False,
        progress_desc: Optional[str] = None
    ) -> Tuple[float, float, Optional[Dict[str, Any]]]:
        
        # For first task or if no stored gradients, use regular training
        G = self.memory.get_matrix()
        if task_id == 0 or G is None:
            return self._train_regular(
                model, optimizer, train_loader, criterion, config,
                task_id, multihead, collect_stats, progress_desc
            )
        
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
        fisher_grad_norms = []
        fisher_proj_grad_norms = []
        eucl_proj_grad_norms = []
        fopng_relative_changes = []  # ||g - P_g|| / ||g||
        
        self._compute_update_prep(F_new, self.F_old, G, config.device)
        iterator = tqdm(train_loader, desc=progress_desc, leave=False) if progress_desc else train_loader
        
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
            
            if collect_stats:
                update, inter_stats = self._compute_update(
                    grad, F_new, self.F_old, G, config.device, return_intermediate=True
                )
                grad_norms.append(inter_stats['raw_grad_norm'])
                fisher_grad_norms.append(inter_stats['fisher_grad_norm'])
                fisher_proj_grad_norms.append(inter_stats['fisher_proj_grad_norm'])
                eucl_proj_grad_norms.append(inter_stats['eucl_proj_grad_norm'])
                update_norms.append(update.norm().item())
                
                # Compute relative change: ||g - P_g|| / ||g||
                P_g = inter_stats['projected_grad']
                diff_norm = (grad - P_g).norm().item()
                raw_norm = inter_stats['raw_grad_norm']
                rel_change = diff_norm / (raw_norm + 1e-10)
                fopng_relative_changes.append(rel_change)
            else:
                update = self._compute_update(grad, F_new, self.F_old, G, config.device)
            
            apply_update(model, update)
            
            total_loss += loss.item() * x.size(0)
            preds = output.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
        
        stats = None
        if collect_stats and grad_norms:
            stats = {
                'grad_norm_mean': np.mean(grad_norms),
                'grad_norm_std': np.std(grad_norms),
                'update_norm_mean': np.mean(update_norms),
                'update_norm_std': np.std(update_norms),
                'fopng_fisher_grad_norm_mean': np.mean(fisher_grad_norms),
                'fopng_fisher_grad_norm_std': np.std(fisher_grad_norms),
                'fopng_fisher_proj_grad_norm_mean': np.mean(fisher_proj_grad_norms),
                'fopng_fisher_proj_grad_norm_std': np.std(fisher_proj_grad_norms),
                'fopng_eucl_proj_grad_norm_mean': np.mean(eucl_proj_grad_norms),
                'fopng_eucl_proj_grad_norm_std': np.std(eucl_proj_grad_norms),
            }
            
            if fopng_relative_changes:
                stats['fopng_projection_relative_change_mean'] = np.mean(fopng_relative_changes)
                stats['fopng_projection_relative_change_std'] = np.std(fopng_relative_changes)
        
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
        collect_stats: bool = False,
        progress_desc: Optional[str] = None
    ) -> Tuple[float, float, Optional[Dict[str, Any]]]:
        """Regular Adam training for first task."""
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        grad_norms = []
        update_norms = []
        
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
        
        if self.F_old is None:
            self.F_old = F_current
        else:
            # Combine Fisher information from old and current tasks
            self.F_old = (self.F_old + F_current) / 2
        
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