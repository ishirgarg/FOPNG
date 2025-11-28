import torch
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Dict, Any, List
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
            if len(self.memory) > 0:
                g_tilde = self.memory.project_orthogonal(g)
                proj_norm = g_tilde.norm().item()
                proj_grad_norms.append(proj_norm)
                proj_to_raw_ratios.append(proj_norm / (raw_norm + 1e-8))
                set_grad_vector(model, g_tilde)
            else:
                proj_grad_norms.append(raw_norm)  # No projection, same as raw
                proj_to_raw_ratios.append(1.0)  # Ratio is 1 when no projection
            
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
        
        # Log average gradient norms and ratios for this epoch (task-specific plots)
        log({
            f"grad_norms_task_{task_id}/ogd_raw_gradient": np.mean(raw_grad_norms),
            f"grad_norms_task_{task_id}/ogd_projected_gradient": np.mean(proj_grad_norms),
            f"grad_ratios_task_{task_id}/ogd_projected_to_raw": np.mean(proj_to_raw_ratios),
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
        lr: float
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute FOPNG update step.
        
        Returns:
            Tuple of (update vector, dict of norms and ratios for logging)
        """
        lam = self.lambda_reg
        norms = {}

        if self.is_diagonal:
            F_new_inv_diag = 1.0 / (F_new + lam)
            F_old_g = F_old * gradient
            G_T_F_old_g = G.T @ F_old_g
            A_inv_G_T_F_old_g = self.A_inv @ G_T_F_old_g
            correction = (G @ A_inv_G_T_F_old_g).view(-1) * F_old.squeeze()
            P_g = gradient - correction
            F_new_inv_P_g = P_g * F_new_inv_diag
            denom = torch.sqrt((P_g * F_new_inv_P_g).sum() + 1e-8)
            v_star = -lr * F_new_inv_P_g / (denom + 1e-8)
            
            # Compute norms and ratios for logging
            raw_norm = gradient.norm().item()
            correction_norm = correction.norm().item()
            v_star_norm = v_star.norm().item()
            
            norms['raw_gradient'] = raw_norm
            norms['correction'] = correction_norm
            norms['v_star'] = v_star_norm
            norms['correction_to_raw_ratio'] = correction_norm / (raw_norm + 1e-8)
            norms['v_star_to_raw_ratio'] = v_star_norm / (raw_norm + 1e-8)
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
            v_star = -lr * F_new_inv_P_g / denom
            
            # Compute norms and ratios for logging
            raw_norm = gradient.norm().item()
            correction_norm = correction.norm().item()
            v_star_norm = v_star.norm().item()
            
            norms['raw_gradient'] = raw_norm
            norms['correction'] = correction_norm
            norms['v_star'] = v_star_norm
            norms['correction_to_raw_ratio'] = correction_norm / (raw_norm + 1e-8)
            norms['v_star_to_raw_ratio'] = v_star_norm / (raw_norm + 1e-8)
        
        return v_star, norms
    
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
        
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        self._compute_update_prep(F_new, self.F_old, G, config.device)
        iterator = tqdm(train_loader, desc=progress_desc, leave=False) if progress_desc else train_loader
        
        # Accumulators for gradient norms and ratios (log average per epoch)
        raw_grad_norms = []
        correction_norms = []
        v_star_norms = []
        correction_to_raw_ratios = []
        v_star_to_raw_ratios = []
        
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
            update, norms = self._compute_update(grad, F_new, self.F_old, G, config.device, config.lr)
            apply_update(model, update)
            
            # Accumulate norms and ratios
            raw_grad_norms.append(norms['raw_gradient'])
            correction_norms.append(norms['correction'])
            v_star_norms.append(norms['v_star'])
            correction_to_raw_ratios.append(norms['correction_to_raw_ratio'])
            v_star_to_raw_ratios.append(norms['v_star_to_raw_ratio'])
            
            total_loss += loss.item() * x.size(0)
            preds = output.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
        
        # Log average gradient norms and ratios for this epoch (task-specific plots)
        log({
            f"grad_norms_task_{task_id}/fopng_raw_gradient": np.mean(raw_grad_norms),
            f"grad_norms_task_{task_id}/fopng_correction": np.mean(correction_norms),
            f"grad_norms_task_{task_id}/fopng_v_star": np.mean(v_star_norms),
            f"grad_ratios_task_{task_id}/fopng_correction_to_raw": np.mean(correction_to_raw_ratios),
            f"grad_ratios_task_{task_id}/fopng_v_star_to_raw": np.mean(v_star_to_raw_ratios),
            "task_id": task_id,
        })
        
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
        
        # Log average gradient norms for this epoch (task-specific plots, no correction/v_star for task 0)
        log({
            f"grad_norms_task_{task_id}/fopng_raw_gradient": np.mean(raw_grad_norms),
            f"grad_norms_task_{task_id}/fopng_correction": 0.0,  # No correction for task 0
            f"grad_norms_task_{task_id}/fopng_v_star": 0.0,  # No v_star for task 0
            f"grad_ratios_task_{task_id}/fopng_correction_to_raw": 0.0,  # No correction for task 0
            f"grad_ratios_task_{task_id}/fopng_v_star_to_raw": 0.0,  # No v_star for task 0
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