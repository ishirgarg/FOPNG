import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


def _cifar_feature_extractor():
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


CIFAR_FEATURE_DIM = 256 * 4 * 4

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
    Lightweight CNN baseline for CIFAR experiments.
    
    Architecture:
        [Conv(3→64) + BN + ReLU] × 2 -> MaxPool
        [Conv(64→128) + BN + ReLU] × 2 -> MaxPool
        [Conv(128→256) + BN + ReLU] × 2 -> MaxPool
        FC 256*4*4 -> 512 -> num_classes
    """
    
    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.features = _cifar_feature_extractor()
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(CIFAR_FEATURE_DIM, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class MultiHeadCIFARCNN(nn.Module):
    """
    CNN trunk shared across tasks with individual heads.
    """
    
    def __init__(
        self,
        num_heads: int,
        head_output_sizes: Optional[List[int]] = None,
        dropout: float = 0.3
    ):
        super().__init__()
        self.features = _cifar_feature_extractor()
        self.shared_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(CIFAR_FEATURE_DIM, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        if head_output_sizes is None:
            head_output_sizes = [2] * num_heads
        
        self.heads = nn.ModuleList([
            nn.Linear(512, out_dim) for out_dim in head_output_sizes
        ])
    
    def forward(self, x, task_id: int = 0):
        x = self.features(x)
        x = self.shared_classifier(x)
        return self.heads[task_id](x)

