from abc import ABC, abstractmethod
from typing import List, Optional
import torch
from torch import nn
from torch.utils.data import DataLoader
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
        device: str
    ) -> torch.Tensor:
        """Estimate Fisher information."""
        pass


class DiagonalFisherEstimator(FisherEstimator):
    """Diagonal approximation of the empirical Fisher."""
    
    def estimate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: str
    ) -> torch.Tensor:
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        model.eval()
        
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data.clone() ** 2
        
        for n in fisher:
            fisher[n] /= len(dataloader)
        
        return torch.cat([fisher[n].view(-1) for n in fisher])


class FullFisherEstimator(FisherEstimator):
    """Full empirical Fisher matrix estimation."""
    
    def estimate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: str
    ) -> torch.Tensor:
        p = get_param_count(model)
        fisher = torch.zeros(p, p, device=device)
        model.eval()
        n_samples = 0
        
        for data, target in dataloader:
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