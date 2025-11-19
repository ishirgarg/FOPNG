from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json
import pickle
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from torch import nn

from config import Config

@dataclass
class EpochLog:
    """Log entry for a single training epoch."""
    task_id: int
    epoch: int
    train_loss: float
    train_acc: float
    grad_norm_mean: Optional[float] = None
    grad_norm_std: Optional[float] = None
    update_norm_mean: Optional[float] = None
    update_norm_std: Optional[float] = None


@dataclass
class EvalLog:
    """Log entry for evaluation after a task."""
    trained_task: int
    eval_task: int
    eval_loss: float
    eval_acc: float


class ExperimentLogger:
    """
    Logger for experiment data, supporting full reproducibility and analysis.
    
    Exports:
    - Raw data (JSON, pickle) for plot reconstruction
    - Plots (PNG, PDF)
    - Model checkpoints (optional)
    - Full configuration and metadata
    """
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        experiment_name: Optional[str] = None,
        config: Optional[Config] = None
    ):
        self.config = config
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup log directory
        if log_dir:
            self.log_dir = Path(log_dir) / self.experiment_name
        else:
            self.log_dir = None
        
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            (self.log_dir / "plots").mkdir(exist_ok=True)
            (self.log_dir / "checkpoints").mkdir(exist_ok=True)
        
        # Data storage
        self.epoch_logs: List[EpochLog] = []
        self.eval_logs: List[EvalLog] = []
        self.results: Dict[int, List[float]] = {}
        self.task_names: Optional[List[str]] = None
        self.method_name: str = ""
        self.dataset_name: str = ""
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Per-epoch detailed stats
        self.detailed_stats: List[Dict[str, Any]] = []
    
    def start_experiment(
        self,
        method_name: str,
        dataset_name: str,
        task_names: Optional[List[str]] = None
    ):
        """Called at experiment start."""
        self.method_name = method_name
        self.dataset_name = dataset_name
        self.task_names = task_names
        self.start_time = datetime.now()
        
        # Reset logs
        self.epoch_logs = []
        self.eval_logs = []
        self.results = {}
        self.detailed_stats = []
    
    def log_epoch(
        self,
        task_id: int,
        epoch: int,
        train_loss: float,
        train_acc: float,
        grad_norm_mean: Optional[float] = None,
        grad_norm_std: Optional[float] = None,
        update_norm_mean: Optional[float] = None,
        update_norm_std: Optional[float] = None,
        extra_stats: Optional[Dict[str, Any]] = None
    ):
        """Log training epoch data."""
        log = EpochLog(
            task_id=task_id,
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            grad_norm_mean=grad_norm_mean,
            grad_norm_std=grad_norm_std,
            update_norm_mean=update_norm_mean,
            update_norm_std=update_norm_std
        )
        self.epoch_logs.append(log)
        
        if extra_stats:
            stat_entry = {
                'task_id': task_id,
                'epoch': epoch,
                **extra_stats
            }
            self.detailed_stats.append(stat_entry)
    
    def log_eval(
        self,
        trained_task: int,
        eval_task: int,
        eval_loss: float,
        eval_acc: float
    ):
        """Log evaluation results."""
        log = EvalLog(
            trained_task=trained_task,
            eval_task=eval_task,
            eval_loss=eval_loss,
            eval_acc=eval_acc
        )
        self.eval_logs.append(log)
    
    def set_results(self, results: Dict[int, List[float]]):
        """Set final results dictionary."""
        self.results = results
    
    def end_experiment(self):
        """Called at experiment end."""
        self.end_time = datetime.now()
    
    def save_model_checkpoint(self, model: nn.Module, name: str = "final"):
        """Save model checkpoint."""
        if self.log_dir:
            path = self.log_dir / "checkpoints" / f"{name}.pt"
            torch.save(model.state_dict(), path)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get experiment metadata."""
        metadata = {
            'experiment_name': self.experiment_name,
            'method_name': self.method_name,
            'dataset_name': self.dataset_name,
            'task_names': self.task_names,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None,
        }
        if self.config:
            metadata['config'] = self.config.to_dict()
        return metadata
    
    def get_raw_data(self) -> Dict[str, Any]:
        """Get all raw data for export."""
        data = {
            'metadata': self.get_metadata(),
            'results': self.results,
            'epoch_logs': [asdict(log) for log in self.epoch_logs],
            'eval_logs': [asdict(log) for log in self.eval_logs],
            'detailed_stats': self.detailed_stats,
        }
        if hasattr(self, 'param_distances'):
            data['param_distances'] = self.param_distances
        return data
    
    def save(self):
        """Save all experiment data to log directory."""
        if not self.log_dir:
            return
        
        raw_data = self.get_raw_data()
        
        # Save as JSON (human-readable)
        json_path = self.log_dir / "experiment_data.json"
        with open(json_path, 'w') as f:
            json.dump(raw_data, f, indent=2, default=str)
        
        # Save as pickle (preserves all Python types)
        pickle_path = self.log_dir / "experiment_data.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(raw_data, f)
        
        # Save results separately for easy loading
        results_path = self.log_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'results': {str(k): v for k, v in self.results.items()},
                'task_names': self.task_names,
                'method_name': self.method_name,
                'dataset_name': self.dataset_name,
            }, f, indent=2)
        
        # Save CSV for spreadsheet analysis
        self._save_csv()
        
        print(f"Experiment data saved to: {self.log_dir}")
    
    def _save_csv(self):
        """Save data in CSV format for easy analysis."""
        if not self.log_dir:
            return
        
        # Epoch logs CSV
        if self.epoch_logs:
            csv_path = self.log_dir / "epoch_logs.csv"
            with open(csv_path, 'w') as f:
                headers = ['task_id', 'epoch', 'train_loss', 'train_acc', 
                          'grad_norm_mean', 'grad_norm_std', 'update_norm_mean', 'update_norm_std']
                f.write(','.join(headers) + '\n')
                for log in self.epoch_logs:
                    values = [str(getattr(log, h, '')) for h in headers]
                    f.write(','.join(values) + '\n')
        
        # Eval logs CSV
        if self.eval_logs:
            csv_path = self.log_dir / "eval_logs.csv"
            with open(csv_path, 'w') as f:
                headers = ['trained_task', 'eval_task', 'eval_loss', 'eval_acc']
                f.write(','.join(headers) + '\n')
                for log in self.eval_logs:
                    values = [str(getattr(log, h, '')) for h in headers]
                    f.write(','.join(values) + '\n')
        
        # Results matrix CSV
        if self.results:
            csv_path = self.log_dir / "accuracy_matrix.csv"
            num_tasks = len(self.results)
            with open(csv_path, 'w') as f:
                # Header: after_task_0, after_task_1, ...
                headers = ['eval_task'] + [f'after_task_{i}' for i in range(num_tasks)]
                f.write(','.join(headers) + '\n')
                for task_id in range(num_tasks):
                    row = [str(task_id)]
                    for i, acc in enumerate(self.results.get(task_id, [])):
                        row.append(f'{acc:.6f}')
                    # Pad with empty for tasks not yet evaluated
                    while len(row) < num_tasks + 1:
                        row.append('')
                    f.write(','.join(row) + '\n')
    
    def save_plot(
        self,
        fig: plt.Figure,
        name: str,
        formats: List[str] = ['png', 'pdf']
    ):
        """Save a matplotlib figure to the plots directory."""
        if not self.log_dir:
            return
        
        for fmt in formats:
            path = self.log_dir / "plots" / f"{name}.{fmt}"
            fig.savefig(path, dpi=150, bbox_inches='tight')
    
    def create_accuracy_plot(self, save: bool = True) -> plt.Figure:
        """Create accuracy progression plot."""
        fig, ax = plt.subplots(figsize=(8, 5))
        
        for task_id, acc_list in sorted(self.results.items()):
            label = self.task_names[task_id] if self.task_names else f"Task {task_id}"
            ax.plot(range(1, len(acc_list) + 1), acc_list, marker='o', label=label)
        
        ax.set_xlabel("After training task k")
        ax.set_ylabel("Accuracy")
        title = f"{self.dataset_name} — {self.method_name}"
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        
        if save and self.log_dir:
            self.save_plot(fig, "accuracy_progression")
        
        return fig
    
    def create_forgetting_plot(self, save: bool = True) -> plt.Figure:
        """Create forgetting visualization."""
        fig, ax = plt.subplots(figsize=(8, 5))
        
        num_tasks = len(self.results)
        forgetting = []
        task_labels = []
        
        for t in range(num_tasks - 1):
            if t in self.results and len(self.results[t]) > 1:
                max_acc = max(self.results[t])
                final_acc = self.results[t][-1]
                forgetting.append((max_acc - final_acc) * 100)
                label = self.task_names[t] if self.task_names else f"Task {t}"
                task_labels.append(label)
        
        if forgetting:
            x = range(len(forgetting))
            ax.bar(x, forgetting, color='coral')
            ax.set_xticks(x)
            ax.set_xticklabels(task_labels, rotation=45, ha='right')
            ax.set_ylabel("Forgetting (%)")
            ax.set_title(f"Forgetting per Task — {self.method_name}")
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            fig.tight_layout()
        
        if save and self.log_dir:
            self.save_plot(fig, "forgetting")
        
        return fig
    
    def create_training_curves_plot(self, save: bool = True) -> plt.Figure:
        """Create training loss/accuracy curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Group by task
        task_epochs = defaultdict(lambda: {'epochs': [], 'loss': [], 'acc': []})
        for log in self.epoch_logs:
            task_epochs[log.task_id]['epochs'].append(log.epoch)
            task_epochs[log.task_id]['loss'].append(log.train_loss)
            task_epochs[log.task_id]['acc'].append(log.train_acc)
        
        # Plot loss
        for task_id, data in sorted(task_epochs.items()):
            label = self.task_names[task_id] if self.task_names else f"Task {task_id}"
            axes[0].plot(data['epochs'], data['loss'], marker='.', label=label)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Training Loss")
        axes[0].set_title("Training Loss per Epoch")
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot accuracy
        for task_id, data in sorted(task_epochs.items()):
            label = self.task_names[task_id] if self.task_names else f"Task {task_id}"
            axes[1].plot(data['epochs'], data['acc'], marker='.', label=label)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Training Accuracy")
        axes[1].set_title("Training Accuracy per Epoch")
        axes[1].legend()
        axes[1].grid(True)
        
        fig.tight_layout()
        
        if save and self.log_dir:
            self.save_plot(fig, "training_curves")
        
        return fig
    
    def create_all_plots(self):
        """Create and save all standard plots."""
        self.create_accuracy_plot(save=True)
        self.create_forgetting_plot(save=True)
        self.create_training_curves_plot(save=True)
        self.create_distribution_drift_plot(save=True)
        plt.close('all')

    def create_distribution_drift_plot(self, save: bool = True) -> plt.Figure:
        """Plot parameter drift over tasks."""
        if not hasattr(self, 'param_distances') or not self.param_distances:
            return None
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        tasks = [d['task'] for d in self.param_distances]
        l2 = [d['l2_distance'] for d in self.param_distances]
        fisher = [d['fisher_distance'] for d in self.param_distances]
        
        ax.plot(tasks, l2, 'o-', label='L2 distance')
        ax.plot(tasks, fisher, 's-', label='Fisher-weighted distance')
        ax.set_xlabel('After training task')
        ax.set_ylabel('Parameter drift from previous task')
        ax.set_title(f'Distribution Change Over Time — {self.method_name}')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        
        if save and self.log_dir:
            self.save_plot(fig, "distribution_drift")
        
        return fig