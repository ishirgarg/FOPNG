from abc import ABC, abstractmethod
from typing import List, Optional
import torch
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad
from torch import nn
from torch.utils.data import DataLoader, Subset
import numpy as np

from utils import get_param_count

class FisherEstimator(ABC):
    """Abstract base class for Fisher information estimation."""
    
    @abstractmethod
    def estimate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: str,
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """Estimate Fisher information. If batch_size is specified, use that instead."""
        pass


class DiagonalFisherEstimator(FisherEstimator):
    """
    True empirical Fisher: average of per-sample squared gradients.
    
    Args:
        use_vmap: If True, use vmap for parallel per-sample gradients (faster but memory-intensive).
                  If False (default), use sequential loop (slower but memory-efficient).
    """
    
    def __init__(self, use_vmap: bool = False):
        self.use_vmap = use_vmap
    
    def estimate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: str,
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        model.eval()
        
        # Clear GPU cache before Fisher estimation to free up memory
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        
        # If batch_size is specified, use only that many samples total for estimation
        if batch_size is not None:
            dataset = dataloader.dataset
            # Take only the first batch_size samples
            limited_dataset = Subset(dataset, range(min(batch_size, len(dataset))))
            fisher_loader = DataLoader(limited_dataset, batch_size=len(limited_dataset), shuffle=False)
        else:
            fisher_loader = dataloader
        
        if self.use_vmap:
            return self._estimate_vmap(model, fisher_loader, criterion, device)
        else:
            return self._estimate_sequential(model, fisher_loader, criterion, device)
    
    def _estimate_sequential(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: str
    ) -> torch.Tensor:
        """Memory-efficient sequential per-sample gradient computation."""
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        total_samples = 0
        
        from tqdm import tqdm
        iterator = tqdm(dataloader, desc="Estimating Fisher", leave=False)
        
        for data, target in iterator:
            data, target = data.to(device), target.to(device)
            
            # Per-sample loop
            for i in range(data.size(0)):
                model.zero_grad()
                output = model(data[i:i+1])
                loss = criterion(output, target[i:i+1])
                loss.backward()
                
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        fisher[n] += p.grad.data ** 2
                
                total_samples += 1
        
        # Average over total samples
        for n in fisher:
            fisher[n] /= total_samples
        
        return torch.cat([fisher[n].view(-1) for n in fisher])
    
    def _estimate_vmap(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: str
    ) -> torch.Tensor:
        """Fast vmap-based per-sample gradient computation (memory-intensive)."""
        # Prepare functional parameters
        params = {name: p for name, p in model.named_parameters()}
        buffers = dict(model.named_buffers())

        # Initialize accumulator
        fisher = {name: torch.zeros_like(p, device=device) 
                  for name, p in params.items()}
        
        total_samples = 0

        # Define a pure function for loss on a SINGLE sample
        def compute_loss_stateless(params, buffers, x, y):
            x_batch = x.unsqueeze(0)
            out = functional_call(model, (params, buffers), (x_batch,))
            out = out.squeeze(0)
            y_batch = y.unsqueeze(0) if y.dim() == 0 else y.unsqueeze(0)
            
            if isinstance(criterion, nn.CrossEntropyLoss):
                 loss = F.cross_entropy(out.unsqueeze(0), y_batch)
            else:
                 loss = criterion(out.unsqueeze(0), y_batch)
            
            return loss

        grad_fn = grad(compute_loss_stateless)

        from tqdm import tqdm
        iterator = tqdm(dataloader, desc="Estimating Fisher (vmap)", leave=False)

        for x, y in iterator:
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)
            total_samples += batch_size

            # Use vmap to compute per-sample gradients in parallel
            batch_grads = vmap(grad_fn, in_dims=(None, None, 0, 0))(params, buffers, x, y)

            # Accumulate squared gradients
            for name, g in batch_grads.items():
                fisher[name] += (g ** 2).sum(dim=0)

        # Divide by N to get the average
        for name in fisher:
            fisher[name] /= total_samples

        return torch.cat([fisher[n].reshape(-1) for n in fisher])


class FullFisherEstimator(FisherEstimator):
    """Full empirical Fisher matrix estimation."""
    
    def estimate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: str,
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        model.eval()
        
        # Clear GPU cache before Fisher estimation to free up memory
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        
        # If batch_size is specified, use only that many samples total for estimation
        if batch_size is not None:
            dataset = dataloader.dataset
            # Take only the first batch_size samples
            limited_dataset = Subset(dataset, range(min(batch_size, len(dataset))))
            fisher_loader = DataLoader(limited_dataset, batch_size=len(limited_dataset), shuffle=False)
        else:
            fisher_loader = dataloader
        
        p = get_param_count(model)
        fisher = torch.zeros(p, p, device=device)
        n_samples = 0
        
        for data, target in fisher_loader:
            data, target = data.to(device), target.to(device)
            for i in range(data.size(0)):
                model.zero_grad()
                output = model(data[i:i+1])
                loss = criterion(output, target[i:i+1])
                loss.backward()
                
                grad = torch.cat([
                    p.grad.view(-1) for p in model.parameters() 
                    if p.grad is not None
                ])
                fisher += torch.outer(grad, grad)
                n_samples += 1
        
        return fisher / n_samples

def fisher_norm_distance(
    model: nn.Module,
    old_params: torch.Tensor,
    new_params: torch.Tensor,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> float:
    """
    Compute Fisher-weighted distance between parameter vectors.
    
    Computes sqrt(d^T F d) where d = new_params - old_params and F is the
    empirical Fisher, without allocating the full Fisher matrix.
    
    Uses identity: d^T F d = (1/N) Σ (d^T g_i)^2
    """
    # Save current params
    saved_params = torch.cat([p.data.view(-1).clone() for p in model.parameters()])
    
    # Set model to old params for Fisher computation
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(old_params[idx:idx+n].view_as(p))
            idx += n
    
    diff = (new_params - old_params).to(device)
    
    model.eval()
    sum_sq_dots = 0.0
    n_samples = 0
    
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        
        for i in range(data.size(0)):
            model.zero_grad()
            output = model(data[i:i+1])
            loss = criterion(output, target[i:i+1])
            loss.backward()
            
            # Get gradient vector
            grad = torch.cat([
                p.grad.view(-1) if p.grad is not None else torch.zeros(p.numel(), device=device)
                for p in model.parameters()
            ])
            
            # Accumulate (d^T g)^2
            dot = torch.dot(diff, grad)
            sum_sq_dots += dot.item() ** 2
            n_samples += 1
    
    # Restore original params
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(saved_params[idx:idx+n].view_as(p))
            idx += n
    
    fisher_dist = np.sqrt(sum_sq_dots / n_samples) if n_samples > 0 else 0.0
    return fisher_dist