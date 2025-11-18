import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class MLP(nn.Module):
    """
    3-layer MLP with:
    - Input: 784 (28x28)
    - Hidden: 100, 100
    - Output: num_classes (default 10)
    """
    
    def __init__(self, input_dim: int = 784, hidden_dim: int = 100, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class MultiHeadMLP(nn.Module):
    """
    Shared MLP trunk with multiple task-specific heads.
    Used for Split MNIST: 5 tasks × 2 classes (default).
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 100,
        num_heads: int = 5,
        head_output_sizes: Optional[List[int]] = None
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        if head_output_sizes is None:
            head_output_sizes = [10] * num_heads
        
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, out_dim) for out_dim in head_output_sizes
        ])
    
    def forward(self, x, task_id: int = 0):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.heads[task_id](x)

