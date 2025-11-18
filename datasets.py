from torch import nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from typing import List, Optional, Tuple

class PermutePixels:
    """Transform that permutes pixels of a (1, 28, 28) tensor."""
    
    def __init__(self, permutation: torch.Tensor):
        self.permutation = permutation
    
    def __call__(self, x):
        c, h, w = x.size()
        flat = x.view(-1)
        permuted = flat[self.permutation]
        return permuted.view(c, h, w)


class FixedRotation:
    """Transform that rotates an image by a fixed angle."""
    
    def __init__(self, angle: float):
        self.angle = angle
    
    def __call__(self, img):
        return TF.rotate(img, self.angle)


class SubsetByClass(Dataset):
    """Wrap a dataset and keep only samples with labels in allowed_classes."""
    
    def __init__(self, base_dataset, allowed_classes):
        self.base_dataset = base_dataset
        self.allowed_classes = set(allowed_classes)
        self.class_to_new = {c: i for i, c in enumerate(sorted(allowed_classes))}
        
        self.indices = [
            i for i, (_, label) in enumerate(base_dataset)
            if int(label) in self.allowed_classes
        ]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        img, label = self.base_dataset[base_idx]
        return img, self.class_to_new[int(label)]


def build_permuted_mnist_tasks(
    num_tasks: int = 5,
    batch_size: int = 10,
    root: str = "./data"
) -> List[Tuple[DataLoader, DataLoader]]:
    """Build Permuted-MNIST tasks."""
    tasks = []
    
    for t in range(num_tasks):
        perm = torch.randperm(28 * 28)
        transform = transforms.Compose([
            transforms.ToTensor(),
            PermutePixels(perm)
        ])
        
        train_ds = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root=root, train=False, download=True, transform=transform)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        tasks.append((train_loader, test_loader))
    
    return tasks


def build_rotated_mnist_tasks(
    angles: Tuple[float, ...] = (0, 10, 20, 30, 40),
    batch_size: int = 10,
    root: str = "./data"
) -> List[Tuple[DataLoader, DataLoader]]:
    """Build Rotated-MNIST tasks."""
    tasks = []
    
    for angle in angles:
        transform = transforms.Compose([
            FixedRotation(angle),
            transforms.ToTensor()
        ])
        
        train_ds = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root=root, train=False, download=True, transform=transform)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        tasks.append((train_loader, test_loader))
    
    return tasks


def build_split_mnist_tasks(
    batch_size: int = 10,
    root: str = "./data"
) -> Tuple[List[Tuple[DataLoader, DataLoader]], List[List[int]]]:
    """Build Split MNIST with 5 tasks, each containing 2 digits."""
    digits_per_task = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    
    base_train = datasets.MNIST(
        root=root, train=True, download=True, transform=transforms.ToTensor()
    )
    base_test = datasets.MNIST(
        root=root, train=False, download=True, transform=transforms.ToTensor()
    )
    
    tasks = []
    for digits in digits_per_task:
        train_subset = SubsetByClass(base_train, digits)
        test_subset = SubsetByClass(base_test, digits)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        tasks.append((train_loader, test_loader))
    
    return tasks, digits_per_task

