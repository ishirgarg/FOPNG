from torch import nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from typing import List, Optional, Tuple

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

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


def _build_cifar_tasks(
    dataset_cls,
    class_groups: List[List[int]],
    batch_size: int,
    root: str,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float]
) -> List[Tuple[DataLoader, DataLoader]]:
    """Generic helper to split CIFAR datasets by class groups."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    base_train = dataset_cls(root=root, train=True, download=True, transform=transform)
    base_test = dataset_cls(root=root, train=False, download=True, transform=transform)
    
    tasks = []
    for classes in class_groups:
        train_subset = SubsetByClass(base_train, classes)
        test_subset = SubsetByClass(base_test, classes)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        tasks.append((train_loader, test_loader))
    
    return tasks


def build_split_cifar10_tasks(
    batch_size: int = 64,
    root: str = "./data/CIFAR",
    class_groups: Optional[List[List[int]]] = None
) -> Tuple[List[Tuple[DataLoader, DataLoader]], List[List[int]]]:
    """
    Build Split CIFAR-10 with 5 tasks of 2 classes each.
    
    Args:
        batch_size: Batch size for the data loaders.
        root: Directory where CIFAR data is stored/downloaded.
        class_groups: Optional manual grouping of classes per task.
    
    Returns:
        tasks: List of (train_loader, test_loader) pairs.
        class_groups: The class splits that were used.
    """
    default_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    class_groups = class_groups or default_groups
    
    tasks = _build_cifar_tasks(
        datasets.CIFAR10,
        class_groups,
        batch_size,
        root,
        CIFAR10_MEAN,
        CIFAR10_STD
    )
    
    return tasks, class_groups


def build_split_cifar100_tasks(
    batch_size: int = 64,
    root: str = "./data/CIFAR",
    num_tasks: int = 10,
    classes_per_task: int = 10,
    class_groups: Optional[List[List[int]]] = None
) -> Tuple[List[Tuple[DataLoader, DataLoader]], List[List[int]]]:
    """
    Build Split CIFAR-100 with 10 tasks of 10 classes each.
    
    Args:
        batch_size: Batch size for the data loaders.
        root: Directory where CIFAR data is stored/downloaded.
        num_tasks: Number of tasks to split into (default 10).
        classes_per_task: Number of classes per task (default 10).
        class_groups: Optional manual grouping of classes per task.
    
    Returns:
        tasks: List of (train_loader, test_loader) pairs.
        class_groups: The class splits that were used.
    """
    if class_groups is None:
        total_classes = num_tasks * classes_per_task
        if total_classes != 100:
            raise ValueError(
                f"num_tasks × classes_per_task must equal 100 for CIFAR-100 (got {total_classes})."
            )
        class_groups = [
            list(range(i * classes_per_task, (i + 1) * classes_per_task))
            for i in range(num_tasks)
        ]
    
    tasks = _build_cifar_tasks(
        datasets.CIFAR100,
        class_groups,
        batch_size,
        root,
        CIFAR100_MEAN,
        CIFAR100_STD
    )
    
    return tasks, class_groups

