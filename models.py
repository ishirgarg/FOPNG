import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


def _cifar_feature_extractor():
    """
    Simplified CNN backbone: 4 convolutional layers.
    Architecture: Conv -> ReLU -> Conv -> ReLU -> MaxPool -> Conv -> ReLU -> Conv -> ReLU -> MaxPool
    """
    return nn.Sequential(
        # Conv block 1: 3 -> 32
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        
        # Conv block 2: 32 -> 64
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


# After 2 maxpools on 32x32: 32 -> 16 -> 8, with 64 channels
CIFAR_FEATURE_DIM = 64 * 8 * 8

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


class SimpleCIFARCNN(nn.Module):
    """
    Simplified CNN for CIFAR experiments.
    
    Architecture:
        4 conv layers (2 blocks with maxpool)
        2 dense layers with dropout
    """
    
    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.features = _cifar_feature_extractor()
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(CIFAR_FEATURE_DIM, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class MultiHeadCIFARCNN(nn.Module):
    """
    Simplified CNN trunk shared across tasks with individual heads.
    4 conv layers + 2 dense layers with dropout.
    """
    
    def __init__(
        self,
        num_heads: int,
        head_output_sizes: Optional[List[int]] = None,
        dropout: float = 0.5
    ):
        super().__init__()
        self.features = _cifar_feature_extractor()
        self.shared_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(CIFAR_FEATURE_DIM, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        if head_output_sizes is None:
            head_output_sizes = [2] * num_heads
        
        self.heads = nn.ModuleList([
            nn.Linear(256, out_dim) for out_dim in head_output_sizes
        ])
    
    def forward(self, x, task_id: int = 0):
        x = self.features(x)
        x = self.shared_classifier(x)
        return self.heads[task_id](x)

