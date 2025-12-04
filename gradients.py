import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import random
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod
from tqdm import tqdm

def get_grad_vector(model: nn.Module) -> torch.Tensor:
    """Concatenate all parameter gradients into a single 1D tensor."""
    grads = []
    for p in model.parameters():
        if p.grad is None:
            grads.append(torch.zeros_like(p.data).view(-1))
        else:
            grads.append(p.grad.view(-1))
    return torch.cat(grads)


def set_grad_vector(model: nn.Module, grad_vector: torch.Tensor):
    """Set model.grad tensors from a single 1D gradient vector."""
    idx = 0
    for p in model.parameters():
        numel = p.data.numel()
        g = grad_vector[idx:idx+numel].view_as(p.data)
        if p.grad is None:
            p.grad = torch.zeros_like(p.data)
        p.grad.copy_(g)
        idx += numel


class GradientMemory:
    """
    Unified gradient memory buffer for continual learning methods.
    
    Supports two modes:
    - 'orthonormal': Store orthonormal basis via Gram-Schmidt (for OGD)
    - 'raw': Store raw gradients as columns of a matrix (for FOPNG)
    
    Both OGD and FOPNG can use the same underlying storage with different
    access patterns.
    """
    
    def __init__(self, mode: str = 'orthonormal', max_directions: int = 2000):
        """
        Args:
            mode: 'orthonormal' for OGD, 'raw' for FOPNG
            max_directions: Maximum number of gradient directions to store
        """
        self.mode = mode
        self.max_directions = max_directions
        self.vectors: List[torch.Tensor] = []
    
    @torch.no_grad()
    def add(self, v: torch.Tensor):
        """Add a gradient direction to memory."""
        if len(self.vectors) >= self.max_directions:
            return
        
        v = v.clone().detach()
        
        if self.mode == 'orthonormal':
            # Gram-Schmidt orthogonalization
            for s in self.vectors:
                proj = torch.dot(v, s) * s
                v = v - proj
            
            norm = torch.norm(v)
            if norm > 1e-8:
                v = v / norm
                self.vectors.append(v)
        else:
            # Raw storage
            self.vectors.append(v)
    
    @torch.no_grad()
    def project_orthogonal(self, g: torch.Tensor) -> torch.Tensor:
        """
        Project g into the subspace orthogonal to stored directions.
        Used by OGD.
        """
        if not self.vectors:
            return g
        
        g_tilde = g.clone()
        for s in self.vectors:
            dot = torch.dot(g_tilde, s)
            g_tilde = g_tilde - dot * s
        return g_tilde
    
    def get_matrix(self) -> Optional[torch.Tensor]:
        """
        Get stored gradients as a matrix (columns are gradients).
        Used by FOPNG.
        """
        if not self.vectors:
            return None
        return torch.stack(self.vectors, dim=1)
    
    def __len__(self) -> int:
        return len(self.vectors)
    
    def clear(self):
        """Clear all stored gradients."""
        self.vectors = []


# =============================================================================
# Gradient Collection Strategies
# =============================================================================

class GradientCollector(ABC):
    """Abstract base class for gradient collection strategies."""
    
    @abstractmethod
    def collect(
        self,
        memory: GradientMemory,
        model: nn.Module,
        dataloader: DataLoader,
        num_directions: int,
        device: str,
        multihead: bool = False,
        task_id: Optional[int] = None
    ):
        """Collect gradient directions from a task."""
        pass
    
    def collect_prefisher(
        self,
        memory: GradientMemory,
        model: nn.Module,
        dataloader: DataLoader,
        num_directions: int,
        fisher_matrix: torch.Tensor,
        device: str,
        multihead: bool = False,
        task_id: Optional[int] = None
    ):
        # Create a temporary memory buffer for raw gradients
        temp_memory = GradientMemory(mode=memory.mode, max_directions=num_directions)
        
        # Collect raw gradients into temporary buffer
        self.collect(temp_memory, model, dataloader, num_directions, device, multihead, task_id)
        
        for grad_vec in temp_memory.vectors:
            if isinstance(fisher_matrix, torch.Tensor):
                # Diagonal Fisher
                if fisher_matrix.dim() == 1:
                    prefisher_grad = fisher_matrix * grad_vec
                else:
                    prefisher_grad = fisher_matrix @ grad_vec
            else:
                prefisher_grad = grad_vec
            memory.add(prefisher_grad)
    
    def collect_empirical_fisher_preconditioned(
        self,
        memory: GradientMemory,
        model: nn.Module,
        dataloader: DataLoader,
        num_directions: int,
        device: str,
        multihead: bool = False,
        task_id: Optional[int] = None
    ):
        """
        Collect gradients pre-multiplied by empirical Fisher using associative property.
        
        Key idea: F = sum_i(g_i * g_i^T), so F*g = sum_i(g_i * (g_i^T * g))
        For each gradient g, we compute F*g by accumulating contributions from all 
        collected gradients without storing the n x n Fisher matrix.
        
        First pass: collect all raw gradients
        Second pass: for each collected gradient, compute F*g and add to memory
        """
        # First pass: collect all raw gradients
        print(f"Collecting empirical Fisher-preconditioned gradients (task {task_id})...")
        temp_memory = GradientMemory(mode=memory.mode, max_directions=num_directions)
        
        self.collect(temp_memory, model, dataloader, num_directions, device, multihead, task_id)
        
        if not temp_memory.vectors:
            return
        
        raw_gradients = temp_memory.vectors  # List of gradient vectors
        
        # Second pass: for each collected gradient, compute F*g on-the-fly
        for g in raw_gradients:
            # Compute F*g = sum_i(g_i * (g_i^T * g))
            Fg = torch.zeros_like(g)
            for g_i in raw_gradients:
                # g_i^T * g: scalar dot product
                dot_prod = torch.dot(g_i, g)
                # g_i * (g_i^T * g): accumulate scaled gradient
                Fg = Fg + g_i * dot_prod
            
            # Add the empirical Fisher preconditioned gradient to memory
            memory.add(Fg)


class GTLCollector(GradientCollector):
    """
    Ground-Truth Logit gradient collector (OGD-GTL).
    Computes gradients with respect to the ground-truth class logit.
    """
    
    def collect(
        self,
        memory: GradientMemory,
        model: nn.Module,
        dataloader: DataLoader,
        num_directions: int,
        device: str,
        multihead: bool = False,
        task_id: Optional[int] = None
    ):
        model.eval()
        collected = 0
        
        desc = "Collecting GTL gradients"
        if task_id is not None:
            desc += f" (task {task_id})"
        iterator = tqdm(dataloader, desc=desc, leave=False)
        
        for x, y in iterator:
            x = x.to(device)
            y = y.to(device)
            
            batch_size = x.size(0)
            for i in range(batch_size):
                if collected >= num_directions:
                    return
                
                model.zero_grad()
                xi = x[i:i+1]
                yi = y[i:i+1]
                
                if multihead:
                    logits = model(xi, task_id=task_id)
                else:
                    logits = model(xi)
                
                # Ground truth logit
                gt_logit = logits[0, yi.item()]
                gt_logit.backward()
                
                grad_vec = get_grad_vector(model).detach()
                memory.add(grad_vec)
                collected += 1
        
        print(f"  Collected {collected} GTL directions (total: {len(memory)})")


class AVECollector(GradientCollector):
    """
    Average logit gradient collector (OGD-AVE).
    Computes gradients with respect to the average of all logits.
    """
    
    def collect(
        self,
        memory: GradientMemory,
        model: nn.Module,
        dataloader: DataLoader,
        num_directions: int,
        device: str,
        multihead: bool = False,
        task_id: Optional[int] = None
    ):
        model.eval()
        collected = 0
        
        desc = "Collecting AVE gradients"
        if task_id is not None:
            desc += f" (task {task_id})"
        iterator = tqdm(dataloader, desc=desc, leave=False)
        
        for x, y in iterator:
            if collected >= num_directions:
                break
            x = x.to(device)
            y = y.to(device)
            
            for i in range(x.size(0)):
                if collected >= num_directions:
                    break
                
                model.zero_grad()
                
                if multihead:
                    output = model(x[i:i+1], task_id=task_id)
                else:
                    output = model(x[i:i+1])
                
                # Average of all logits
                avg_logit = output.mean()
                avg_logit.backward()
                
                grad_vec = get_grad_vector(model).detach()
                memory.add(grad_vec)
                collected += 1
        
        print(f"  Collected {collected} AVE directions (total: {len(memory)})")

