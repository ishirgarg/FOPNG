from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import json
import torch
from torch import nn
import numpy as np

from utils import load_experiment

def plot_results(
    results: Dict[int, List[float]],
    title: str = "Continual Learning Results",
    task_names: Optional[List[str]] = None
):
    """Plot accuracy progression for each task."""
    plt.figure(figsize=(8, 5))
    
    for task_id, acc_list in sorted(results.items()):
        label = task_names[task_id] if task_names else f"Task {task_id}"
        plt.plot(range(1, len(acc_list) + 1), acc_list, marker='o', label=label)
    
    plt.xlabel("After training task k")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compare_experiments(
    experiment_paths: List[Union[str, Path]],
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Compare multiple experiments in a single plot.
    
    Args:
        experiment_paths: List of paths to experiment log directories
        save_path: Optional path to save the comparison plot
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for exp_path in experiment_paths:
        data = load_experiment(exp_path)
        results = data['results']
        method = data['metadata']['method_name']
        dataset = data['metadata']['dataset_name']
        
        # Compute average accuracy across tasks
        num_tasks = len(results)
        avg_accs = []
        for after_task in range(num_tasks):
            accs = [results[str(t)][after_task] if after_task < len(results[str(t)]) else None 
                   for t in range(after_task + 1)]
            accs = [a for a in accs if a is not None]
            avg_accs.append(np.mean(accs) if accs else 0)
        
        label = f"{method} ({dataset})"
        ax.plot(range(1, num_tasks + 1), avg_accs, marker='o', label=label)
    
    ax.set_xlabel("After training task k")
    ax.set_ylabel("Average Accuracy")
    ax.set_title("Comparison of Continual Learning Methods")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig