"""
Wandb-based logging system for experiments.

Usage:
    # Initialize once at the start of your experiment
    from logger import init_wandb, log
    
    init_wandb(project="my-project", name="exp-1", config=config)
    
    # Log metrics from anywhere - step auto-increments
    log({"train_loss": 0.5, "train_acc": 0.9})
    log({"eval/accuracy": 0.85})
    
    # Or use the ExperimentLogger for structured logging
    logger = ExperimentLogger(config, log_dir, "method", "dataset")
    logger.log_epoch(task_id=0, epoch=1, train_loss=0.5, train_acc=0.9)
"""

# Export get_step for consistent step access
__all__ = ['init_wandb', 'log', 'get_step', 'finish', 'ExperimentLogger']

import wandb
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from config import Config


# Global logger instance - initialized by init_wandb()
_wandb_run: Optional[wandb.run] = None
_global_step: int = 0  # Single global step counter - always increments


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
    global _wandb_run, _global_step
    
    if not use_wandb:
        _wandb_run = None
        _global_step = 0
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
    
    _global_step = 0
    
    # Define custom x-axes for meaningful chart displays
    # Training metrics use task_id as x-axis (which task we're training)
    wandb.define_metric("task_id")
    wandb.define_metric("train/*", step_metric="task_id")
    
    # Evaluation metrics use trained_task as x-axis (how many tasks trained so far)
    wandb.define_metric("trained_task")
    wandb.define_metric("accuracy_progression/*", step_metric="trained_task")
    wandb.define_metric("loss/*", step_metric="trained_task")
    wandb.define_metric("param_drift/*", step_metric="trained_task")
    wandb.define_metric("average_accuracy_test", step_metric="trained_task")
    wandb.define_metric("average_accuracy_train", step_metric="trained_task")
    
    # Per-batch metrics use global_batch_idx as x-axis (monotonic across epochs/tasks)
    wandb.define_metric("global_batch_idx")
    wandb.define_metric("fopng_batch/*", step_metric="global_batch_idx")
    
    print(f"Wandb initialized: {_wandb_run.url}")


def log(metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
    """
    Log metrics to wandb using the global step counter.
    
    The step counter always increments to ensure monotonically increasing steps.
    
    Args:
        metrics: Dictionary of metric names to values
        step: Ignored - always uses global counter for consistency
        commit: Whether to commit this log entry (default True)
    
    Examples:
        from logger import log
        log({"my_metric": value})  # Uses next global step
    """
    global _global_step
    
    if _wandb_run is None:
        return
    
    _wandb_run.log(metrics, step=_global_step, commit=commit)
    _global_step += 1


def get_step() -> int:
    """Get the current global step (for reference/logging purposes)."""
    return _global_step


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
        self.results: Dict[int, Dict[str, List[float]]] = {}  # {task_id: {'test': [...], 'train': [...]}}
        self.epoch_logs: List[Dict[str, Any]] = []  # Per-epoch training logs
        self.eval_logs: List[Dict[str, Any]] = []  # Evaluation logs
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
        })
    
    def log_epoch(
        self,
        task_id: int,
        epoch: int,
        train_loss: float,
        train_acc: float
    ):
        """Log training epoch data - logs directly to wandb and stores locally."""
        global _global_step
        
        # Log directly to wandb (log() auto-increments global step)
        metrics = {
            f"train/task_{task_id}/loss": train_loss,
            f"train/task_{task_id}/accuracy": train_acc,
            "task_id": task_id,
            "epoch": epoch,
        }
        
        current_step = _global_step
        log(metrics)
        
        # Store locally for CSV export
        self.epoch_logs.append({
            'task_id': task_id,
            'epoch': epoch,
            'step': current_step,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
        })
    
    def log_eval(
        self,
        trained_task: int,
        eval_task: int,
        eval_loss: float,
        eval_acc: float,
        train_loss: float,
        train_acc: float
    ):
        """Log evaluation results - logs directly to wandb and stores locally."""
        global _global_step
        
        # Log accuracy and loss metrics (log() auto-increments global step)
        metrics = {
            f"accuracy_progression/task_{eval_task}_test": eval_acc,
            f"accuracy_progression/task_{eval_task}_train": train_acc,
            f"loss/task_{eval_task}_test": eval_loss,
            f"loss/task_{eval_task}_train": train_loss,
            "trained_task": trained_task,
            "eval_task": eval_task,
        }
        
        # Store results for plot generation (separate train/test)
        if eval_task not in self.results:
            self.results[eval_task] = {'test': [], 'train': []}
        self.results[eval_task]['test'].append(eval_acc)
        self.results[eval_task]['train'].append(train_acc)
        
        # Compute and log average accuracy of all tasks trained so far
        # Average test accuracy across all tasks that have been evaluated at this stage
        all_test_accs = []
        all_train_accs = []
        for task_id in range(trained_task + 1):
            if task_id in self.results:
                test_accs = self.results[task_id].get('test', []) if isinstance(self.results[task_id], dict) else self.results[task_id]
                train_accs = self.results[task_id].get('train', []) if isinstance(self.results[task_id], dict) else []
                
                # Only include the latest accuracy for this task (at current trained_task stage)
                if test_accs and len(test_accs) > 0:
                    all_test_accs.append(test_accs[-1])
                if train_accs and len(train_accs) > 0:
                    all_train_accs.append(train_accs[-1])
        
        if all_test_accs:
            metrics["average_accuracy_test"] = np.mean(all_test_accs)
        if all_train_accs:
            metrics["average_accuracy_train"] = np.mean(all_train_accs)
        
        current_step = _global_step
        log(metrics)
        
        # Store locally for CSV export
        self.eval_logs.append({
            'trained_task': trained_task,
            'eval_task': eval_task,
            'step': current_step,
            'test_loss': eval_loss,
            'test_accuracy': eval_acc,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
        })
    
    def set_results(self, results: Dict[int, List[float]]):
        """Set final results dictionary (for backward compatibility).
        
        This method expects the old format with only test accuracies.
        It will convert it to the new format with separate train/test data.
        Note: This will LOSE any train accuracy data that was previously logged.
        Prefer not to call this method and let log_eval populate results instead.
        """
        # Convert old format (test-only) to new format
        for task_id, test_accs in results.items():
            if task_id not in self.results:
                self.results[task_id] = {'test': [], 'train': []}
            # Only update test accs if not already populated
            if not self.results[task_id].get('test'):
                self.results[task_id]['test'] = test_accs
    
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
        formats: List[str] = ['png', 'pdf'],
        upload_to_wandb: bool = True
    ):
        """Save a matplotlib figure locally and optionally to wandb."""
        if not self.log_dir:
            return
        
        for fmt in formats:
            path = self.log_dir / "plots" / f"{name}.{fmt}"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            
            # Log to wandb as image (only if requested)
            if _wandb_run is not None and fmt == 'png' and upload_to_wandb:
                wandb.log({f"plots/{name}": wandb.Image(fig)})
    
    def create_accuracy_plot(self, save: bool = True) -> plt.Figure:
        """Create accuracy progression plot showing per-task test accuracy vs tasks trained."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot one line per task, showing how its test accuracy changes as more tasks are trained
        for task_id in sorted(self.results.keys()):
            acc_list = self.results[task_id].get('test', []) if isinstance(self.results[task_id], dict) else self.results[task_id]
            label = self.task_names[task_id] if self.task_names else f"Task {task_id}"
            # x-axis: number of tasks trained so far (1, 2, 3, ...)
            # y-axis: accuracy on this task
            x_values = range(1, len(acc_list) + 1)
            ax.plot(x_values, acc_list, marker='o', linewidth=2, markersize=6, label=label)
        
        ax.set_xlabel("After training task k", fontsize=12)
        ax.set_ylabel("Test Accuracy", fontsize=12)
        title = f"{self.dataset_name} — {self.method_name} (Test)"
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        if save:
            self.save_plot(fig, "accuracy_progression_test", upload_to_wandb=True)
        
        return fig
    
    def create_train_accuracy_plot(self, save: bool = True) -> plt.Figure:
        """Create accuracy progression plot showing per-task train accuracy vs tasks trained."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot one line per task, showing how its train accuracy changes as more tasks are trained
        for task_id in sorted(self.results.keys()):
            acc_list = self.results[task_id].get('train', []) if isinstance(self.results[task_id], dict) else []
            if not acc_list:
                continue
            label = self.task_names[task_id] if self.task_names else f"Task {task_id}"
            # x-axis: number of tasks trained so far (1, 2, 3, ...)
            # y-axis: accuracy on this task
            x_values = range(1, len(acc_list) + 1)
            ax.plot(x_values, acc_list, marker='s', linewidth=2, markersize=6, label=label)
        
        ax.set_xlabel("After training task k", fontsize=12)
        ax.set_ylabel("Train Accuracy", fontsize=12)
        title = f"{self.dataset_name} — {self.method_name} (Train)"
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        if save:
            self.save_plot(fig, "accuracy_progression_train", upload_to_wandb=True)
        
        return fig
    
    def create_forgetting_plot(self, save: bool = True) -> plt.Figure:
        """Create forgetting visualization for test accuracy."""
        fig, ax = plt.subplots(figsize=(8, 5))
        
        num_tasks = len(self.results)
        forgetting = []
        task_labels = []
        
        for t in range(num_tasks - 1):
            if t in self.results:
                acc_data = self.results[t].get('test', []) if isinstance(self.results[t], dict) else self.results[t]
                if len(acc_data) > 1:
                    max_acc = max(acc_data)
                    final_acc = acc_data[-1]
                    forgetting.append((max_acc - final_acc) * 100)
                    label = self.task_names[t] if self.task_names else f"Task {t}"
                    task_labels.append(label)
        
        if forgetting:
            x = range(len(forgetting))
            ax.bar(x, forgetting, color='coral')
            ax.set_xticks(x)
            ax.set_xticklabels(task_labels, rotation=45, ha='right')
            ax.set_ylabel("Forgetting (%)")
            ax.set_title(f"Test Forgetting per Task — {self.method_name}")
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            fig.tight_layout()
        
        if save:
            self.save_plot(fig, "forgetting_test", upload_to_wandb=True)
        
        return fig
    
    def create_train_forgetting_plot(self, save: bool = True) -> plt.Figure:
        """Create forgetting visualization for train accuracy."""
        fig, ax = plt.subplots(figsize=(8, 5))
        
        num_tasks = len(self.results)
        forgetting = []
        task_labels = []
        
        for t in range(num_tasks - 1):
            if t in self.results:
                acc_data = self.results[t].get('train', []) if isinstance(self.results[t], dict) else []
                if len(acc_data) > 1:
                    max_acc = max(acc_data)
                    final_acc = acc_data[-1]
                    forgetting.append((max_acc - final_acc) * 100)
                    label = self.task_names[t] if self.task_names else f"Task {t}"
                    task_labels.append(label)
        
        if forgetting:
            x = range(len(forgetting))
            ax.bar(x, forgetting, color='lightblue')
            ax.set_xticks(x)
            ax.set_xticklabels(task_labels, rotation=45, ha='right')
            ax.set_ylabel("Forgetting (%)")
            ax.set_title(f"Train Forgetting per Task — {self.method_name}")
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            fig.tight_layout()
        
        if save:
            self.save_plot(fig, "forgetting_train", upload_to_wandb=True)
        
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
    
    def log_summary_metrics(self):
        """Log summary metrics to wandb as raw data for interactive plots."""
        if _wandb_run is None:
            return
        
        num_tasks = len(self.results)
        if num_tasks == 0:
            return
        
        # Collect all summary metrics in one dict and log once
        summary_metrics = {}
        
        # Compute forgetting metrics
        test_forgetting_list = []
        train_forgetting_list = []
        
        for t in range(num_tasks - 1):
            if t in self.results:
                # Test forgetting
                test_accs = self.results[t].get('test', []) if isinstance(self.results[t], dict) else self.results[t]
                if len(test_accs) > 1:
                    test_forgetting = (max(test_accs) - test_accs[-1]) * 100
                    test_forgetting_list.append(test_forgetting)
                    summary_metrics[f"forgetting_bar/test_task_{t}"] = test_forgetting
                
                # Train forgetting
                train_accs = self.results[t].get('train', []) if isinstance(self.results[t], dict) else []
                if len(train_accs) > 1:
                    train_forgetting = (max(train_accs) - train_accs[-1]) * 100
                    train_forgetting_list.append(train_forgetting)
                    summary_metrics[f"forgetting_bar/train_task_{t}"] = train_forgetting
        
        # Average forgetting metrics
        if test_forgetting_list:
            summary_metrics["summary/avg_test_forgetting"] = np.mean(test_forgetting_list)
        if train_forgetting_list:
            summary_metrics["summary/avg_train_forgetting"] = np.mean(train_forgetting_list)
        
        # Final accuracies for each task
        final_test_accs = []
        final_train_accs = []
        for t in range(num_tasks):
            if t in self.results:
                test_accs = self.results[t].get('test', []) if isinstance(self.results[t], dict) else self.results[t]
                train_accs = self.results[t].get('train', []) if isinstance(self.results[t], dict) else []
                
                if test_accs:
                    summary_metrics[f"final_accuracy_bar/test_task_{t}"] = test_accs[-1]
                    final_test_accs.append(test_accs[-1])
                if train_accs:
                    summary_metrics[f"final_accuracy_bar/train_task_{t}"] = train_accs[-1]
                    final_train_accs.append(train_accs[-1])
        
        # Average final accuracies
        if final_test_accs:
            summary_metrics["summary/avg_final_test_accuracy"] = np.mean(final_test_accs)
        if final_train_accs:
            summary_metrics["summary/avg_final_train_accuracy"] = np.mean(final_train_accs)
        
        # Parameter drift summary if available
        if hasattr(self, 'param_distances') and self.param_distances:
            avg_l2 = np.mean([d['l2_distance'] for d in self.param_distances])
            avg_fisher_train = np.mean([d['fisher_distance_train'] for d in self.param_distances])
            avg_fisher_test = np.mean([d['fisher_distance_test'] for d in self.param_distances])
            
            summary_metrics["summary/avg_l2_distance"] = avg_l2
            summary_metrics["summary/avg_fisher_distance_train"] = avg_fisher_train
            summary_metrics["summary/avg_fisher_distance_test"] = avg_fisher_test
        
        # Log all summary metrics in one call (uses global step)
        if summary_metrics:
            log(summary_metrics)
    
    def create_all_plots(self):
        """Create and save all standard plots locally and upload key plots to wandb."""
        # Create and upload accuracy progression plots (line charts showing task accuracy over time)
        self.create_accuracy_plot(save=True)  # Test accuracy
        self.create_train_accuracy_plot(save=True)  # Train accuracy
        
        # Create and upload forgetting bar charts
        self.create_forgetting_plot(save=True)  # Test forgetting
        self.create_train_forgetting_plot(save=True)  # Train forgetting
        
        # Distribution drift plot
        self.create_distribution_drift_plot(save=True)
        
        # Log summary metrics to wandb
        self.log_summary_metrics()
        
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
        
        # Save epoch logs as CSV
        if self.epoch_logs:
            epoch_csv_path = self.log_dir / "epoch_logs.csv"
            with open(epoch_csv_path, 'w') as f:
                f.write('task_id,epoch,step,train_loss,train_accuracy\n')
                for log in self.epoch_logs:
                    f.write(f"{log['task_id']},{log['epoch']},{log['step']},{log['train_loss']:.6f},{log['train_accuracy']:.6f}\n")
        
        # Save eval logs as CSV
        if self.eval_logs:
            eval_csv_path = self.log_dir / "eval_logs.csv"
            with open(eval_csv_path, 'w') as f:
                f.write('trained_task,eval_task,step,test_loss,test_accuracy,train_loss,train_accuracy\n')
                for log in self.eval_logs:
                    f.write(f"{log['trained_task']},{log['eval_task']},{log['step']},{log['test_loss']:.6f},{log['test_accuracy']:.6f},{log['train_loss']:.6f},{log['train_accuracy']:.6f}\n")
        
        # Save test accuracy matrix CSV
        if self.results:
            csv_path = self.log_dir / "accuracy_matrix.csv"
            num_tasks = len(self.results)
            with open(csv_path, 'w') as f:
                headers = ['eval_task'] + [f'after_task_{i}' for i in range(num_tasks)]
                f.write(','.join(headers) + '\n')
                for task_id in range(num_tasks):
                    row = [str(task_id)]
                    acc_data = self.results.get(task_id, {})
                    test_accs = acc_data.get('test', []) if isinstance(acc_data, dict) else acc_data
                    for acc in test_accs:
                        row.append(f'{acc:.6f}')
                    while len(row) < num_tasks + 1:
                        row.append('')
                    f.write(','.join(row) + '\n')
            
            # Save train accuracy matrix CSV
            csv_train_path = self.log_dir / "train_accuracy_matrix.csv"
            with open(csv_train_path, 'w') as f:
                headers = ['eval_task'] + [f'after_task_{i}' for i in range(num_tasks)]
                f.write(','.join(headers) + '\n')
                for task_id in range(num_tasks):
                    row = [str(task_id)]
                    acc_data = self.results.get(task_id, {})
                    train_accs = acc_data.get('train', []) if isinstance(acc_data, dict) else []
                    for acc in train_accs:
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
            'epoch_logs': self.epoch_logs,
            'eval_logs': self.eval_logs,
        }
        if hasattr(self, 'param_distances'):
            data['param_distances'] = self.param_distances
        if hasattr(self, 'train_results'):
            data['train_results'] = self.train_results
        return data
