"""
Wandb-based logging system for experiments.

Usage:
    # Initialize once at the start of your experiment
    from logger import init_wandb, log
    
    init_wandb(project="my-project", name="exp-1", config=config)
    
    # Log metrics from anywhere in your code
    log({"train_loss": 0.5, "train_acc": 0.9}, step=100)
    log({"eval/accuracy": 0.85}, step=200)
    
    # Or use the logger instance directly
    from logger import wandb_logger
    wandb_logger.log({"metric": value})
"""

# Export _max_epochs_per_task for step calculation consistency
__all__ = ['init_wandb', 'log', 'set_step', 'set_task_epoch', 'finish', 'ExperimentLogger', '_max_epochs_per_task']

import wandb
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from torch import nn

from config import Config


# Global logger instance - initialized by init_wandb()
_wandb_run: Optional[wandb.run] = None
_current_step: int = 0
_task_id: int = 0
_epoch: int = 0
_max_epochs_per_task: int = 1000  # Used for step calculation - will be updated from config


def init_wandb(
    project: str = "fopng-experiments",
    name: Optional[str] = None,
    config: Optional[Config] = None,
    entity: Optional[str] = None,
    tags: Optional[List[str]] = None,
    resume: Optional[Union[str, bool]] = None,
    use_wandb: bool = True,
    **kwargs
):
    """
    Initialize wandb logging.
    
    Args:
        project: Wandb project name
        name: Run name (defaults to timestamp)
        config: Config object to log as wandb config
        entity: Wandb entity/team name
        tags: List of tags for the run
        resume: Resume a previous run (see wandb.init docs)
        use_wandb: Whether to actually initialize wandb (False = no-op)
        **kwargs: Additional arguments passed to wandb.init
    """
    global _wandb_run, _current_step
    
    if not use_wandb:
        _wandb_run = None
        _current_step = 0
        return
    
    if _wandb_run is not None:
        print("Warning: wandb already initialized. Reinitializing...")
        wandb.finish()
    
    # Convert config to dict if provided
    wandb_config = {}
    if config:
        wandb_config = config.to_dict()
    
    # Generate run name if not provided
    if name is None:
        name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize wandb
    _wandb_run = wandb.init(
        project=project,
        name=name,
        config=wandb_config,
        entity=entity,
        tags=tags,
        resume=resume,
        **kwargs
    )
    
    _current_step = 0
    print(f"Wandb initialized: {_wandb_run.url}")


def log(metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
    """
    Log metrics to wandb. Can be called from anywhere in the code.
    
    This is the simplest way to log metrics - just import and call:
        from logger import log
        log({"my_metric": value})
    
    Args:
        metrics: Dictionary of metric names to values
        step: Step number (defaults to internal counter)
        commit: Whether to commit this log entry (default True)
    
    Examples:
        # Basic usage - logs to wandb automatically
        log({"train_loss": 0.5, "train_acc": 0.9})
        
        # With explicit step
        log({"eval/accuracy": 0.85}, step=100)
        
        # Log from anywhere - optimizers, utils, etc.
        from logger import log
        log({"custom_metric": value, "another_metric": value2})
    """
    global _current_step
    
    if _wandb_run is None:
        # Silently ignore if wandb is disabled
        return
    
    if step is None:
        step = _current_step
    
    _wandb_run.log(metrics, step=step, commit=commit)
    
    if step is None:
        _current_step += 1


def set_step(step: int):
    """Set the current step counter."""
    global _current_step
    _current_step = step


def set_task_epoch(task_id: int, epoch: int):
    """Set current task and epoch for convenience."""
    global _task_id, _epoch
    _task_id = task_id
    _epoch = epoch


def finish():
    """Finish the wandb run."""
    global _wandb_run
    if _wandb_run is not None:
        wandb.finish()
        _wandb_run = None


# Backward compatibility: ExperimentLogger class that wraps wandb
class ExperimentLogger:
    """
    Logger for experiment data using wandb.
    
    This class maintains backward compatibility with the old interface
    while using wandb under the hood.
    """
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        experiment_name: Optional[str] = None,
        config: Optional[Config] = None,
        project: str = "fopng-experiments",
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **wandb_kwargs
    ):
        self.config = config
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(log_dir) / self.experiment_name if log_dir else None
        
        # Check if wandb should be used
        use_wandb = getattr(config, 'use_wandb', True) if config else True
        
        # Initialize wandb if not already initialized
        global _wandb_run
        if _wandb_run is None and use_wandb:
            init_wandb(
                project=project,
                name=self.experiment_name,
                config=config,
                entity=entity,
                tags=tags,
                use_wandb=use_wandb,
                **wandb_kwargs
            )
        
        # Minimal storage - only what's needed for plots
        self.results: Dict[int, List[float]] = {}  # Only for generating plots
        self.task_names: Optional[List[str]] = None
        self.method_name: str = ""
        self.dataset_name: str = ""
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Create log directory for checkpoints/plots if needed
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            (self.log_dir / "plots").mkdir(exist_ok=True)
            (self.log_dir / "checkpoints").mkdir(exist_ok=True)
    
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
        
        # Reset results (only thing we store for plots)
        self.results = {}
        
        # Log experiment metadata (only essential info, not config parameters)
        log({
            "experiment/method": method_name,
            "experiment/dataset": dataset_name,
        }, step=0)
        
        # Set max epochs per task from config if available
        global _max_epochs_per_task
        if self.config and hasattr(self.config, 'epochs_per_task'):
            # Use a multiplier that's at least 1000 to ensure room for epochs
            _max_epochs_per_task = max(1000, self.config.epochs_per_task * 2)
    
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
        """Log training epoch data - logs directly to wandb, no storage."""
        # Log directly to wandb - only log loss and accuracy, no grad norms
        metrics = {
            f"train/task_{task_id}/loss": train_loss,
            f"train/task_{task_id}/accuracy": train_acc,
        }
        
        # Use monotonic step counter: task_id * max_epochs + epoch
        # This ensures steps are always increasing
        global _max_epochs_per_task
        step = task_id * _max_epochs_per_task + epoch
        log(metrics, step=step)
    
    def log_eval(
        self,
        trained_task: int,
        eval_task: int,
        eval_loss: float,
        eval_acc: float,
        train_loss: float,
        train_acc: float
    ):
        """Log evaluation results - logs directly to wandb and updates results for plots."""
        # Log directly to wandb - log task accuracy as function of tasks trained
        # Use the step of the last epoch of the trained task
        # Last epoch step = task_id * max_epochs + epochs_per_task
        global _max_epochs_per_task
        epochs_per_task = self.config.epochs_per_task if self.config and hasattr(self.config, 'epochs_per_task') else 1
        step = trained_task * _max_epochs_per_task + epochs_per_task
        
        # Log to training/ section: task accuracy as function of tasks trained
        # trained_task + 1 represents how many tasks have been trained so far
        metrics = {
            f"train/task_{eval_task}/accuracy_vs_tasks_trained": eval_acc,
        }
        log(metrics, step=step)
        
        # Only store results for plot generation (minimal storage)
        if eval_task not in self.results:
            self.results[eval_task] = []
        self.results[eval_task].append(eval_acc)
    
    def set_results(self, results: Dict[int, List[float]]):
        """Set final results dictionary."""
        self.results = results
        
        # Don't log final accuracies separately - they're already logged in log_eval
    
    def end_experiment(self):
        """Called at experiment end."""
        self.end_time = datetime.now()
        
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            log({
                "experiment/duration_seconds": duration,
            })
    
    def save_model_checkpoint(self, model: nn.Module, name: str = "final"):
        """Save model checkpoint (both locally and to wandb)."""
        if self.log_dir:
            path = self.log_dir / "checkpoints" / f"{name}.pt"
            torch.save(model.state_dict(), path)
            
            # Also save to wandb
            if _wandb_run is not None:
                wandb.save(str(path))
    
    def save_plot(
        self,
        fig: plt.Figure,
        name: str,
        formats: List[str] = ['png', 'pdf']
    ):
        """Save a matplotlib figure (both locally and to wandb)."""
        if not self.log_dir:
            return
        
        for fmt in formats:
            path = self.log_dir / "plots" / f"{name}.{fmt}"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            
            # Log to wandb
            if _wandb_run is not None and fmt == 'png':
                wandb.log({f"plots/{name}": wandb.Image(fig)})
    
    def create_accuracy_plot(self, save: bool = True) -> plt.Figure:
        """Create accuracy progression plot showing per-task accuracy vs tasks trained."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot one line per task, showing how its accuracy changes as more tasks are trained
        for task_id, acc_list in sorted(self.results.items()):
            label = self.task_names[task_id] if self.task_names else f"Task {task_id}"
            # x-axis: number of tasks trained so far (1, 2, 3, ...)
            # y-axis: accuracy on this task
            x_values = range(1, len(acc_list) + 1)
            ax.plot(x_values, acc_list, marker='o', linewidth=2, markersize=6, label=label)
        
        ax.set_xlabel("After training task k", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        title = f"{self.dataset_name} — {self.method_name}"
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        if save:
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
        
        if save:
            self.save_plot(fig, "forgetting")
        
        return fig
    
    def create_training_curves_plot(self, save: bool = True) -> plt.Figure:
        """Create merged plot showing task accuracy as function of tasks trained for all tasks."""
        # Build data structure: for each task, track accuracy vs tasks trained
        # results[task_id] = [acc_after_task_0, acc_after_task_1, ...]
        
        if not self.results:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot one line per task, showing how its accuracy changes as more tasks are trained
        for task_id, acc_list in sorted(self.results.items()):
            label = self.task_names[task_id] if self.task_names else f"Task {task_id}"
            # x-axis: number of tasks trained so far (1, 2, 3, ...)
            # y-axis: accuracy on this task
            x_values = range(1, len(acc_list) + 1)
            ax.plot(x_values, acc_list, marker='o', linewidth=2, markersize=6, label=label)
        
        ax.set_xlabel("After training task k", fontsize=12)
        ax.set_ylabel("Task Accuracy", fontsize=12)
        title = f"{self.dataset_name} — {self.method_name}"
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        if save:
            self.save_plot(fig, "training_task_accuracy_vs_tasks_trained")
        
        return fig
    
    def create_distribution_drift_plot(self, save: bool = True) -> plt.Figure:
        """Plot parameter drift over tasks."""
        if not hasattr(self, 'param_distances') or not self.param_distances:
            return None
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        tasks = [d['task'] for d in self.param_distances]
        l2 = [d['l2_distance'] for d in self.param_distances]
        fisher_train = [d['fisher_distance_train'] for d in self.param_distances]
        fisher_test = [d['fisher_distance_test'] for d in self.param_distances]
        
        ax.plot(tasks, l2, 'o-', label='L2 distance')
        ax.plot(tasks, fisher_train, 's-', label='Fisher distance (train)')
        ax.plot(tasks, fisher_test, '^-', label='Fisher distance (test)')
        ax.set_xlabel('After training task')
        ax.set_ylabel('Parameter drift from previous task')
        ax.set_title(f'Distribution Change Over Time — {self.method_name}')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        
        if save:
            self.save_plot(fig, "distribution_drift")
        
        return fig
    
    def create_all_plots(self):
        """Create and save all standard plots."""
        self.create_accuracy_plot(save=True)
        self.create_forgetting_plot(save=True)
        # Create merged plot for training task accuracy vs tasks trained
        self.create_training_curves_plot(save=True)
        self.create_distribution_drift_plot(save=True)
        plt.close('all')
    
    def save(self):
        """Save minimal experiment data to log directory (for backward compatibility)."""
        if not self.log_dir:
            return
        
        raw_data = self.get_raw_data()
        
        # Save as JSON
        json_path = self.log_dir / "experiment_data.json"
        with open(json_path, 'w') as f:
            json.dump(raw_data, f, indent=2, default=str)
        
        # Save as pickle
        pickle_path = self.log_dir / "experiment_data.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(raw_data, f)
        
        # Save results separately
        results_path = self.log_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'results': {str(k): v for k, v in self.results.items()},
                'task_names': self.task_names,
                'method_name': self.method_name,
                'dataset_name': self.dataset_name,
            }, f, indent=2)
        
        # Save results matrix CSV (only thing we have data for)
        if self.results:
            csv_path = self.log_dir / "accuracy_matrix.csv"
            num_tasks = len(self.results)
            with open(csv_path, 'w') as f:
                headers = ['eval_task'] + [f'after_task_{i}' for i in range(num_tasks)]
                f.write(','.join(headers) + '\n')
                for task_id in range(num_tasks):
                    row = [str(task_id)]
                    for i, acc in enumerate(self.results.get(task_id, [])):
                        row.append(f'{acc:.6f}')
                    while len(row) < num_tasks + 1:
                        row.append('')
                    f.write(','.join(row) + '\n')
        
        print(f"Experiment data saved to: {self.log_dir}")
    
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
        """Get minimal raw data for export (most data is in wandb)."""
        data = {
            'metadata': self.get_metadata(),
            'results': self.results,
        }
        if hasattr(self, 'param_distances'):
            data['param_distances'] = self.param_distances
        if hasattr(self, 'train_results'):
            data['train_results'] = self.train_results
        return data
