from abc import ABC, abstractmethod
from typing import List, Optional
import torch
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad
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
        task_id: Optional[int] = None
    ) -> torch.Tensor:
        model.eval()
        
        # Clear GPU cache before Fisher estimation to free up memory
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        
        if self.use_vmap:
            return self._estimate_vmap(model, dataloader, criterion, device)
        else:
            return self._estimate_sequential(model, dataloader, criterion, device)
    
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
        task_id: Optional[int] = None
    ) -> torch.Tensor:
        model.eval()
        
        # Clear GPU cache before Fisher estimation to free up memory
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        
        p = get_param_count(model)
        fisher = torch.zeros(p, p, device=device)
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
        Compute A and G factors using explicit gradient computation.
        
        Args:
            max_samples: Maximum number of samples to use (default 1000).
                        Use fewer samples for speed vs. full dataset for accuracy.
        """
        print(f"[KFAC] Starting Fisher estimation (max {max_samples} samples)...")
        
        # Storage
        A_sum = {}
        G_sum = {}
        n_samples = 0
        
        # Identify Linear layers
        layer_names = []
        layer_modules = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                layer_names.append(name)
                layer_modules[name] = module
                A_sum[name] = None
                G_sum[name] = None
                self.fisher_factors[name] = {'A': None, 'G': None}
        
        if not layer_names:
            print(f"[KFAC] Warning: No Linear layers found in model!")
            print(f"[KFAC] Model structure: {[n for n, m in model.named_modules()]}")
            return self.fisher_factors
        
        print(f"[KFAC] Registered layers: {layer_names}")
        
        was_training = model.training
        model.train()
        
        batch_idx = 0
        for data, target in dataloader:
            if n_samples >= max_samples:
                break
                
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            
            # Process each sample individually
            for i in range(batch_size):
                if n_samples >= max_samples:
                    break
                
                # Single sample
                x_i = data[i:i+1]
                
                # === SINGLE FORWARD PASS: Capture both activations and outputs ===
                activations = {}
                layer_outputs = {}
                
                def save_activation(name):
                    def hook(module, input, output):
                        # Save input to layer (this is 'a' in the paper)
                        # Don't detach - we need it in the computation graph for gradient computation
                        act = input[0].view(-1)  # Flatten, keep in graph
                        # Add bias term
                        act_with_bias = torch.cat([act, torch.ones(1, device=act.device, requires_grad=False)])
                        activations[name] = act_with_bias
                    return hook
                
                def save_output(name):
                    def hook(module, input, output):
                        # Save layer output (pre-activation for Linear layers)
                        layer_outputs[name] = output
                    return hook
                
                # Register both hooks
                handles = []
                for name in layer_names:
                    handles.append(layer_modules[name].register_forward_hook(save_activation(name)))
                    handles.append(layer_modules[name].register_forward_hook(save_output(name)))
                
                # Single forward pass
                model.zero_grad()
                if task_id is not None:
                    output = model(x_i, task_id=task_id)
                else:
                    output = model(x_i)
                
                # Sample from model distribution
                with torch.no_grad():
                    if isinstance(criterion, nn.CrossEntropyLoss):
                        probs = torch.softmax(output, dim=1)
                        sampled_target = torch.multinomial(probs, 1).squeeze(1)  # Remove last dim: [1,1] -> [1]
                    else:
                        sampled_target = output.detach()
                
                # Compute loss
                loss = criterion(output, sampled_target)
                
                # Compute gradients of loss w.r.t. each layer output
                for name in layer_names:
                    if name in layer_outputs and name in activations:
                        layer_out = layer_outputs[name]
                        
                        # Compute ∂L/∂layer_out
                        grad_out = torch.autograd.grad(
                            outputs=loss,
                            inputs=layer_out,
                            retain_graph=True,
                            create_graph=False,
                            only_inputs=True
                        )[0]
                        
                        # g_i = ∂L/∂s (gradient w.r.t. layer output)
                        g_i = grad_out.detach().view(-1)  # Shape: (out_dim,)
                        
                        # a_i = input with bias (detach for accumulation)
                        a_i = activations[name].detach()  # Shape: (in_dim + 1,)
                        
                        # Accumulate A = Σ a_i a_i^T
                        A_sample = torch.outer(a_i, a_i)
                        if A_sum[name] is None:
                            A_sum[name] = A_sample
                        else:
                            A_sum[name] += A_sample
                        
                        # Accumulate G = Σ g_i g_i^T
                        G_sample = torch.outer(g_i, g_i)
                        if G_sum[name] is None:
                            G_sum[name] = G_sample
                        else:
                            G_sum[name] += G_sample
                
                # Cleanup
                for handle in handles:
                    handle.remove()
                
                n_samples += 1
                
            batch_idx += 1
            if batch_idx % 50 == 0:
                print(f"[KFAC] Batch {batch_idx}, samples: {n_samples}...")
        
        model.train(was_training)
        
        # Normalize and store
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
                    self.fisher_factors[name] = {'A': A_new, 'G': G_new}
            else:
                print(f"[KFAC] Warning: No data for layer {name}")
                if name in self.fisher_factors:
                    del self.fisher_factors[name]
        
        self.step_count += 1
        self.inverses_valid = False
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
        Uses exact per-sample formula: A = E[ā_i ā_i^T], G = E[g_i g_i^T]
        """
        activations = {}
        per_sample_grad_accum = {}
        
        # Hook setup
        def save_activation(name):
            def hook(module, input, output):
                act = input[0].detach().view(input[0].size(0), -1)
                ones = torch.ones(act.size(0), 1, device=act.device)
                activations[name] = torch.cat([act, ones], dim=1)
            return hook
        
        # Register forward hooks
        handles = []
        layer_names = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(save_activation(name)))
                layer_names.append(name)
                if name not in self.fisher_factors:
                    self.fisher_factors[name] = {'A': None, 'G': None}
        
        # Forward pass
        was_training = model.training
        model.train()
        
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
        
        # Compute per-sample losses
        batch_size = data.size(0)
        if isinstance(criterion, nn.CrossEntropyLoss):
            per_sample_criterion = nn.CrossEntropyLoss(reduction='none')
            per_sample_losses = per_sample_criterion(output, sampled_targets)
        else:
            per_sample_losses = criterion(output, sampled_targets)
            if per_sample_losses.dim() == 0:
                per_sample_losses = per_sample_losses.unsqueeze(0).expand(batch_size)
        
        # Register backward hooks for per-sample gradients
        grad_handles = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in layer_names:
                def make_grad_hook(layer_name):
                    def hook(module, grad_input, grad_output):
                        g = grad_output[0].detach().view(grad_output[0].size(0), -1)
                        if g.size(0) == 1:
                            g_i = g[0:1]
                            if layer_name not in per_sample_grad_accum:
                                per_sample_grad_accum[layer_name] = g_i.t() @ g_i
                            else:
                                per_sample_grad_accum[layer_name] = per_sample_grad_accum[layer_name] + g_i.t() @ g_i
                    return hook
                grad_handles.append(module.register_full_backward_hook(make_grad_hook(name)))
        
        # Backward per-sample
        for i in range(batch_size):
            model.zero_grad()
            per_sample_losses[i].backward(retain_graph=(i < batch_size - 1))
        
        # Remove gradient hooks
        for handle in grad_handles:
            handle.remove()
        
        # Update running averages with per-sample statistics
        for name in layer_names:
            if name in activations:
                act = activations[name]  # (batch_size, in_dim+1)
                # A_batch = (1/batch_size) * Σ_i act_i act_i^T
                A_batch = (act.t() @ act) / batch_size
                
                if name in per_sample_grad_accum:
                    # G_batch = (1/batch_size) * Σ_i g_i g_i^T
                    G_batch = per_sample_grad_accum[name] / batch_size
                    
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
            
            # Check condition numbers
            A_cond = torch.linalg.cond(A_damped).item()
            G_cond = torch.linalg.cond(G_damped).item()
            print(f"[{name}] A condition: {A_cond:.2e}, G condition: {G_cond:.2e}")
            if A_cond > 1e6 or G_cond > 1e6:
                print(f"  WARNING: Poorly conditioned matrices!")
            
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
                # Remove bias row/col from A_inv
                A_inv_weight = A_inv[:-1, :-1]
                # Apply: G^{-1} @ grad @ A^{-1}
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
            
            # Check condition numbers
            A_cond = torch.linalg.cond(A_damped).item()
            G_cond = torch.linalg.cond(G_damped).item()
            print(f"[{name}] A condition: {A_cond:.2e}, G condition: {G_cond:.2e}")
            if A_cond > 1e6 or G_cond > 1e6:
                print(f"  WARNING: Poorly conditioned matrices!")
            
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