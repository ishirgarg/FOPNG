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
        device: str,
        task_id: Optional[int] = None
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
        device: str,
        task_id: Optional[int] = None
    ) -> torch.Tensor:
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        model.eval()
        
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            if task_id is not None:
                output = model(data, task_id=task_id)
            else:
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
        device: str,
        task_id: Optional[int] = None
    ) -> torch.Tensor:
        p = get_param_count(model)
        fisher = torch.zeros(p, p, device=device)
        model.eval()
        n_samples = 0
        
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            for i in range(data.size(0)):
                model.zero_grad()
                if task_id is not None:
                    output = model(data[i:i+1], task_id=task_id)
                else:
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


class KFACFisherEstimator(FisherEstimator):
    """KFAC block-diagonal Fisher approximation with moving averages."""
    
    def __init__(self, epsilon=0.95, use_running_avg=True, inversion_freq=None):
        super().__init__()
        self.fisher_factors = {}  # Running averages of A and G
        self.epsilon = epsilon  # Moving average decay parameter
        self.use_running_avg = use_running_avg  # Whether to use incremental updates
        self.step_count = 0  # Track number of updates
        self.inversion_freq = inversion_freq  # How often to recompute inverses (None = manual)
        self.cached_inverses = {}  # Cached inverse factors
        self.inverses_valid = False  # Whether cached inverses are up-to-date
        self.last_inversion_step = -1  # Last step when inverses were computed
    
    def estimate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: str,
        task_id: Optional[int] = None,
        max_samples: int = 1000
    ) -> dict:
        """
        Compute A and G factors for each layer.
        
        Args:
            max_samples: Maximum number of samples to use (default 1000).
                        Use fewer samples for speed vs. full dataset for accuracy.
        """
        print(f"[KFAC] Starting Fisher estimation (max {max_samples} samples)...")
        activations = {}
        pre_activation_grads = {}
        
        # ============================================================
        # HOOKS: Capture ā (forward) and g (backward)
        # ============================================================
        def save_activation(name):
            def hook(module, input, output):
                act = input[0].detach().view(input[0].size(0), -1)
                # Add bias term: ā = [a; 1]
                ones = torch.ones(act.size(0), 1, device=act.device)
                activations[name] = torch.cat([act, ones], dim=1)
            return hook
        
        def save_pre_activation_grad(name):
            def hook(module, grad_input, grad_output):
                # grad_output[0] = ∂L/∂s (pre-activation gradient)
                g = grad_output[0].detach().view(grad_output[0].size(0), -1)
                pre_activation_grads[name] = g
            return hook
        
        # Register hooks on Linear layers
        handles = []
        layer_names = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(save_activation(name)))
                handles.append(module.register_full_backward_hook(save_pre_activation_grad(name)))
                layer_names.append(name)
                self.fisher_factors[name] = {'A': None, 'G': None}
        
        # Use train mode to ensure hooks fire properly
        was_training = model.training
        model.train()
        
        # ============================================================
        # ACCUMULATE: Compute E[ā ā^T] and E[g g^T]
        # ============================================================
        A_sum = {name: None for name in layer_names}
        G_sum = {name: None for name in layer_names}
        n_samples = 0
        
        batch_count = 0
        for data, target in dataloader:
            batch_count += 1
            if batch_count % 50 == 0:
                print(f"[KFAC] Batch {batch_count}, samples: {n_samples}...")
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            
            # Early exit if we've collected enough samples
            if n_samples >= max_samples:
                print(f"[KFAC] Reached max_samples ({max_samples}), stopping early")
                break
            
            # Forward pass (handle multihead models)
            model.zero_grad()
            if task_id is not None:
                output = model(data, task_id=task_id)
            else:
                output = model(data)
            
            # Sample from model distribution (Section 5 of paper)
            with torch.no_grad():
                if isinstance(criterion, nn.CrossEntropyLoss):
                    probs = torch.softmax(output, dim=1)
                    sampled_targets = torch.multinomial(probs, 1).squeeze()
                else:
                    sampled_targets = output.detach()
            
            # Backward pass
            loss = criterion(output, sampled_targets)
            loss.backward()
            
            # Accumulate outer products: A = Σ ā ā^T, G = Σ g g^T
            for name in layer_names:
                if name in activations and name in pre_activation_grads:
                    act = activations[name]
                    g = pre_activation_grads[name]
                    
                    if A_sum[name] is None:
                        A_sum[name] = act.t() @ act
                        G_sum[name] = g.t() @ g
                    else:
                        A_sum[name] = A_sum[name] + act.t() @ act
                        G_sum[name] = G_sum[name] + g.t() @ g
            
            n_samples += batch_size
            activations.clear()
            pre_activation_grads.clear()
        
        # Cleanup and normalize
        for handle in handles:
            handle.remove()
        
        # Restore original training state
        model.train(was_training)
        
        for name in layer_names:
            if A_sum[name] is not None and G_sum[name] is not None:
                A_new = A_sum[name] / n_samples
                G_new = G_sum[name] / n_samples
                
                if self.use_running_avg and name in self.fisher_factors and \
                   self.fisher_factors[name]['A'] is not None:
                    # Update with moving average: A = ε*A_old + (1-ε)*A_new
                    self.fisher_factors[name]['A'] = (
                        self.epsilon * self.fisher_factors[name]['A'] + 
                        (1 - self.epsilon) * A_new
                    )
                    self.fisher_factors[name]['G'] = (
                        self.epsilon * self.fisher_factors[name]['G'] + 
                        (1 - self.epsilon) * G_new
                    )
                else:
                    # First time or not using running average: initialize
                    self.fisher_factors[name]['A'] = A_new
                    self.fisher_factors[name]['G'] = G_new
            else:
                # Hooks didn't capture (e.g., unused heads in multihead models)
                # Skip this layer - don't add to fisher_factors
                print(f"[KFAC] Warning: No activations captured for layer {name}")
                if name in self.fisher_factors:
                    del self.fisher_factors[name]
        
        self.step_count += 1
        self.inverses_valid = False  # Invalidate cached inverses
        avg_type = "moving average" if self.use_running_avg else "batch estimate"
        print(f"[KFAC] Fisher estimation complete ({len(self.fisher_factors)} layers, {avg_type})")
        return self.fisher_factors
    
    def update_running_average(
        self,
        model: nn.Module,
        data: torch.Tensor,
        target: torch.Tensor,
        criterion: nn.Module,
        device: str,
        task_id: Optional[int] = None
    ):
        """
        Incrementally update running averages of A and G with a single mini-batch.
        This is the proper online K-FAC update for use during training.
        """
        activations = {}
        pre_activation_grads = {}
        
        # Hook setup
        def save_activation(name):
            def hook(module, input, output):
                act = input[0].detach().view(input[0].size(0), -1)
                ones = torch.ones(act.size(0), 1, device=act.device)
                activations[name] = torch.cat([act, ones], dim=1)
            return hook
        
        def save_pre_activation_grad(name):
            def hook(module, grad_input, grad_output):
                g = grad_output[0].detach().view(grad_output[0].size(0), -1)
                pre_activation_grads[name] = g
            return hook
        
        # Register hooks
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(save_activation(name)))
                handles.append(module.register_full_backward_hook(save_pre_activation_grad(name)))
                if name not in self.fisher_factors:
                    self.fisher_factors[name] = {'A': None, 'G': None}
        
        # Forward + backward pass
        was_training = model.training
        model.train()
        model.zero_grad()
        
        if task_id is not None:
            output = model(data, task_id=task_id)
        else:
            output = model(data)
        
        # Sample from model distribution
        with torch.no_grad():
            if isinstance(criterion, nn.CrossEntropyLoss):
                probs = torch.softmax(output, dim=1)
                sampled_targets = torch.multinomial(probs, 1).squeeze()
            else:
                sampled_targets = output.detach()
        
        loss = criterion(output, sampled_targets)
        loss.backward()
        
        # Update running averages
        batch_size = data.size(0)
        for name in list(activations.keys()):
            if name in activations and name in pre_activation_grads:
                act = activations[name]
                g = pre_activation_grads[name]
                
                # Compute batch statistics
                A_batch = (act.t() @ act) / batch_size
                G_batch = (g.t() @ g) / batch_size
                
                if self.fisher_factors[name]['A'] is None:
                    # First time: initialize
                    self.fisher_factors[name]['A'] = A_batch
                    self.fisher_factors[name]['G'] = G_batch
                else:
                    # Moving average: A = ε*A_old + (1-ε)*A_batch
                    self.fisher_factors[name]['A'] = (
                        self.epsilon * self.fisher_factors[name]['A'] + 
                        (1 - self.epsilon) * A_batch
                    )
                    self.fisher_factors[name]['G'] = (
                        self.epsilon * self.fisher_factors[name]['G'] + 
                        (1 - self.epsilon) * G_batch
                    )
        
        # Cleanup
        for handle in handles:
            handle.remove()
        model.train(was_training)
        
        self.step_count += 1
        self.inverses_valid = False  # Invalidate cached inverses when A and G change
        
        # Optionally recompute inverses if frequency is set
        if self.inversion_freq is not None and \
           self.step_count - self.last_inversion_step >= self.inversion_freq:
            self.recompute_inverses(damping=1e-3)
    
    def recompute_inverses(self, damping=1e-3):
        """
        Explicitly recompute and cache inverse factors.
        Should be called after A and G have been updated significantly.
        """
        print(f"[KFAC] Recomputing inverses (step {self.step_count})...")
        self.cached_inverses = self._compute_inverses(damping)
        self.inverses_valid = True
        self.last_inversion_step = self.step_count
    
    def get_inverse_factors(self, damping=1e-3):
        """
        Get inverse factors. Uses cache if valid, otherwise recomputes.
        
        Note: For efficiency, explicitly call recompute_inverses() when needed
        rather than relying on automatic recomputation.
        """
        if not self.inverses_valid or not self.cached_inverses:
            self.recompute_inverses(damping)
        return self.cached_inverses
    
    def _compute_inverses(self, damping=1e-3):
        """Internal method to compute Ā^{-1} and G^{-1} with Tikhonov damping."""
        inverse_factors = {}
        sqrt_damping = torch.sqrt(torch.tensor(damping))
        
        for name, factors in self.fisher_factors.items():
            A, G = factors['A'], factors['G']
            
            # Move to CPU for linear algebra operations if on MPS (better compatibility)
            device = A.device
            use_cpu = str(device).startswith('mps')
            
            if use_cpu:
                A = A.cpu()
                G = G.cpu()
            
            # Factored damping (Section 6.3)
            # π_i = √(tr(Ā)/dim(Ā) / tr(G)/dim(G))
            pi = torch.sqrt((torch.trace(A) / A.size(0)) / 
                           (torch.trace(G) / G.size(0) + 1e-8))
            
            # Ã = Ā + π√λ I,  G̃ = G + (1/π)√λ I
            A_damped = A + pi * sqrt_damping * torch.eye(A.size(0), device=A.device)
            G_damped = G + (1/pi) * sqrt_damping * torch.eye(G.size(0), device=G.device)
            
            # Invert (use Cholesky for stability)
            try:
                A_chol = torch.linalg.cholesky(A_damped)
                A_inv = torch.cholesky_inverse(A_chol)
                
                G_chol = torch.linalg.cholesky(G_damped)
                G_inv = torch.cholesky_inverse(G_chol)
            except Exception as e:
                A_inv = torch.linalg.inv(A_damped)
                G_inv = torch.linalg.inv(G_damped)
            
            # Move back to original device if needed
            if use_cpu:
                A_inv = A_inv.to(device)
                G_inv = G_inv.to(device)
            
            inverse_factors[name] = (A_inv, G_inv)
        
        return inverse_factors
    
    def apply_inverse(self, gradients, damping=1e-3):
        """
        Apply preconditioner: U = G^{-1} · grad · A^{-1}
        
        Args:
            gradients: Dict {layer_name: weight_gradient_matrix}
        Returns:
            Dict of preconditioned gradients
        """
        inverse_factors = self.get_inverse_factors(damping)
        preconditioned = {}
        
        for name, grad in gradients.items():
            if name in inverse_factors:
                A_inv, G_inv = inverse_factors[name]
                
                # Handle bias: A_inv is (in_dim+1)×(in_dim+1), grad is (out_dim)×(in_dim)
                A_inv_weight = A_inv[:-1, :-1]  # Remove bias row/col
                
                # Precondition: G^{-1} · grad · A^{-1}
                temp = grad @ A_inv_weight
                preconditioned[name] = G_inv @ temp
            else:
                preconditioned[name] = grad
        
        return preconditioned
    
    def apply_fisher(self, gradients):
        """
        Apply Fisher matrix: F @ grad = (G ⊗ A) @ grad = G · grad · A^T
        
        Args:
            gradients: Dict {layer_name: weight_gradient_matrix}
        Returns:
            Dict with Fisher applied
        """
        result = {}
        
        for name, grad in gradients.items():
            if name in self.fisher_factors:
                A = self.fisher_factors[name]['A']
                G = self.fisher_factors[name]['G']
                
                # Handle bias: A is (in_dim+1)×(in_dim+1), grad is (out_dim)×(in_dim)
                A_weight = A[:-1, :-1]  # Remove bias row/col
                
                # Apply Fisher: G · grad · A^T
                temp = grad @ A_weight.T
                result[name] = G @ temp
            else:
                result[name] = grad
        
        return result

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