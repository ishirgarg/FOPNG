from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import pickle
import json
import torch
from torch import nn
import numpy as np
import random
from torch.utils.data import DataLoader

def load_experiment(path: Union[str, Path]) -> Dict[str, Any]:
    """Load experiment data from a log directory."""
    path = Path(path)
    
    # Try pickle first (preserves all types)
    pickle_path = path / "experiment_data.pkl"
    if pickle_path.exists():
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    
    # Fall back to JSON
    json_path = path / "experiment_data.json"
    if json_path.exists():
        with open(json_path, 'r') as f:
            return json.load(f)
    
    raise FileNotFoundError(f"No experiment data found in {path}")

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.mps, 'manual_seed'):
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_update(model: nn.Module, update: torch.Tensor):
    """Apply a parameter update vector to model parameters."""
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.add_(update[idx:idx+n].view_as(p))
            idx += n


def get_param_count(model: nn.Module) -> int:
    """Get total number of parameters in model."""
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    multihead: bool = False,
    task_id: Optional[int] = None
) -> Tuple[float, float]:
    """Evaluate model on a dataloader. Returns (loss, accuracy)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        
        if multihead:
            logits = model(x, task_id=task_id)
        else:
            logits = model(x)
        
        loss = criterion(logits, y)
        preds = logits.argmax(dim=1)
        
        total_loss += loss.item() * x.size(0)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)
    
    return total_loss / total_samples, total_correct / total_samples