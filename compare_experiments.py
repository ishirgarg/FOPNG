#!/usr/bin/env python3
"""
compare_experiments.py

Load saved experiment outputs (JSON/PKL) and generate comprehensive comparison plots.

Features:
- Load experiment data from directories (tries utils.load_experiment)
- Multiple plot types:
  * Mean accuracy over time
  * Final per-task accuracies (grouped bars)
  * Forgetting per task (grouped bars)
  * Accuracy matrix heatmaps
  * Training curves (loss and accuracy)
  * Parameter drift (L2 and Fisher distances)
  * Backward transfer
  * Forward transfer
- CLI: pass multiple experiment paths with optional labels
- Generate all plots or select specific ones

Example:
  python compare_experiments.py \
    ./experiments/ogd_gtl_permuted ./experiments/fopng_diagonal_permuted \
    --labels "OGD-GTL","FOPNG" --out-dir ./comparisons --show

  # Generate only specific plots
  python compare_experiments.py exp1 exp2 exp3 \
    --plots mean_accuracy forgetting training_curves --show

  # Generate all plots
  python compare_experiments.py exp1 exp2 --plots all --out-dir ./all_plots
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

try:
    from utils import load_experiment
except Exception:
    # Fallback: simple loader
    def load_experiment(p: Path):
        p = Path(p)
        if p.is_dir():
            j = p / "experiment_data.json"
            if j.exists():
                return json.loads(j.read_text())
            r = p / "results.json"
            if r.exists():
                return json.loads(r.read_text())
        if p.exists():
            return json.loads(p.read_text())
        raise FileNotFoundError(f"No experiment data found at {p}")


def extract_results(data: Dict) -> Dict[int, List[float]]:
    """Return results mapping: task_id -> list of accuracies."""
    if 'results' in data and isinstance(data['results'], dict):
        return {int(k): v for k, v in data['results'].items()}
    
    if all(isinstance(k, str) and isinstance(v, list) for k, v in data.items()):
        try:
            return {int(k): v for k, v in data.items()}
        except ValueError:
            pass
    
    raise ValueError('Unrecognized experiment file format')


def extract_epoch_logs(data: Dict) -> List[Dict]:
    """Extract epoch logs if available."""
    if 'epoch_logs' in data:
        return data['epoch_logs']
    return []


def extract_param_distances(data: Dict) -> List[Dict]:
    """Extract parameter distance data if available."""
    if 'param_distances' in data:
        return data['param_distances']
    return []


def compute_summary(results: Dict[int, List[float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (mean_over_time, final_accs, forgetting)."""
    if not results:
        return np.array([]), np.array([]), np.array([])
    
    num_tasks = max(results.keys()) + 1
    task_lists = [results.get(t, []) for t in range(num_tasks)]
    
    # Mean accuracy over time
    mean_over_time = []
    for k in range(1, num_tasks + 1):
        vals = []
        for t in range(k):
            lst = task_lists[t]
            if len(lst) >= k:
                vals.append(lst[k - 1])
        mean_over_time.append(np.mean(vals) if vals else np.nan)
    
    # Final accuracies and forgetting
    final_accs = np.array([task_lists[t][-1] if task_lists[t] else np.nan for t in range(num_tasks)])
    max_accs = np.array([max(task_lists[t]) if task_lists[t] else np.nan for t in range(num_tasks)])
    forgetting = max_accs - final_accs
    
    return np.array(mean_over_time), final_accs, forgetting


def compute_backward_transfer(results: Dict[int, List[float]]) -> np.ndarray:
    """Compute backward transfer for each task.
    
    BWT_i = (1/(i-1)) * sum_{j<i} (acc_i[j] - acc_j[j])
    where acc_i[j] is accuracy on task j after training task i.
    """
    if not results:
        return np.array([])
    
    num_tasks = max(results.keys()) + 1
    bwt = []
    
    for i in range(1, num_tasks):
        transfer_sum = 0.0
        count = 0
        for j in range(i):
            if j in results and len(results[j]) > i and len(results[j]) > j:
                acc_after_i = results[j][i]
                acc_after_j = results[j][j]
                transfer_sum += (acc_after_i - acc_after_j)
                count += 1
        bwt.append(transfer_sum / count if count > 0 else 0.0)
    
    return np.array(bwt)


def compute_forward_transfer(results: Dict[int, List[float]]) -> np.ndarray:
    """Compute forward transfer for each task.
    
    FWT_i = acc_0[i] - random_baseline
    where acc_0[i] is accuracy on task i before any training (requires zero-shot eval).
    
    Since we don't typically have zero-shot, we approximate as the initial accuracy
    improvement on a task compared to a random baseline.
    """
    # This is a simplified version - true forward transfer requires zero-shot evaluation
    return np.array([])


def create_accuracy_matrix(results: Dict[int, List[float]]) -> np.ndarray:
    """Create accuracy matrix where entry (i, j) is accuracy on task i after training task j."""
    if not results:
        return np.array([])
    
    num_tasks = max(results.keys()) + 1
    matrix = np.full((num_tasks, num_tasks), np.nan)
    
    for task_id, acc_list in results.items():
        for trained_after, acc in enumerate(acc_list):
            if trained_after < num_tasks:
                matrix[task_id, trained_after] = acc * 100  # Convert to percentage
    
    return matrix


# =============================================================================
# Plot Functions
# =============================================================================

def plot_mean_accuracy(experiments: List[Tuple[str, Dict[int, List[float]]]], 
                       out: Optional[Path] = None, show: bool = False):
    """Plot mean accuracy over time."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for label, results in experiments:
        mean_ot, _, _ = compute_summary(results)
        if len(mean_ot) > 0:
            x = np.arange(1, len(mean_ot) + 1)
            ax.plot(x, mean_ot, marker='o', label=label, linewidth=2)
    
    ax.set_xlabel('After training task k', fontsize=11)
    ax.set_ylabel('Mean accuracy across seen tasks', fontsize=11)
    ax.set_title('Mean Accuracy Over Time', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    
    _save_and_show(fig, out, show, 'mean_accuracy')


def plot_final_accuracies(experiments: List[Tuple[str, Dict[int, List[float]]]], 
                          out: Optional[Path] = None, show: bool = False):
    """Plot final per-task accuracies as grouped bars."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    n_exps = len(experiments)
    summaries = [(label, *compute_summary(results)) for label, results in experiments]
    max_num_tasks = max(len(s[2]) for s in summaries) if summaries else 0
    
    indices = np.arange(max_num_tasks)
    width = 0.8 / max(1, n_exps)
    
    for i, (label, _, final_accs, _) in enumerate(summaries):
        vals = final_accs if final_accs.size else np.full(max_num_tasks, np.nan)
        if vals.size < max_num_tasks:
            vals = np.concatenate([vals, np.full(max_num_tasks - vals.size, np.nan)])
        ax.bar(indices + i * width, vals * 100.0, width=width, label=label)
    
    ax.set_xlabel('Task ID', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Final Per-Task Accuracies', fontsize=13, fontweight='bold')
    ax.set_xticks(indices + width * (n_exps - 1) / 2)
    ax.set_xticklabels([str(i) for i in range(max_num_tasks)])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    
    _save_and_show(fig, out, show, 'final_accuracies')


def plot_forgetting(experiments: List[Tuple[str, Dict[int, List[float]]]], 
                   out: Optional[Path] = None, show: bool = False):
    """Plot forgetting per task as grouped bars."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    n_exps = len(experiments)
    summaries = [(label, *compute_summary(results)) for label, results in experiments]
    max_num_tasks = max(len(s[3]) for s in summaries) if summaries else 0
    
    indices = np.arange(max_num_tasks)
    width = 0.8 / max(1, n_exps)
    
    for i, (label, _, _, forgetting) in enumerate(summaries):
        vals = forgetting if forgetting.size else np.full(max_num_tasks, np.nan)
        if vals.size < max_num_tasks:
            vals = np.concatenate([vals, np.full(max_num_tasks - vals.size, np.nan)])
        ax.bar(indices + i * width, vals * 100.0, width=width, label=label)
    
    ax.set_xlabel('Task ID', fontsize=11)
    ax.set_ylabel('Forgetting (%)', fontsize=11)
    ax.set_title('Forgetting Per Task', fontsize=13, fontweight='bold')
    ax.set_xticks(indices + width * (n_exps - 1) / 2)
    ax.set_xticklabels([str(i) for i in range(max_num_tasks)])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)
    fig.tight_layout()
    
    _save_and_show(fig, out, show, 'forgetting')


def plot_accuracy_matrices(experiments: List[Tuple[str, Dict[int, List[float]]]], 
                           out: Optional[Path] = None, show: bool = False):
    """Plot accuracy matrices as heatmaps."""
    n_exps = len(experiments)
    ncols = min(3, n_exps)
    nrows = (n_exps + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if n_exps == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (label, results) in enumerate(experiments):
        matrix = create_accuracy_matrix(results)
        if matrix.size == 0:
            continue
        
        ax = axes[idx]
        im = ax.imshow(matrix, cmap='YlGnBu', aspect='auto', vmin=0, vmax=100)
        
        # Add text annotations
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if not np.isnan(matrix[i, j]):
                    text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                                 ha="center", va="center", color="black", fontsize=8)
        
        ax.set_xlabel('After training task', fontsize=10)
        ax.set_ylabel('Evaluated on task', fontsize=10)
        ax.set_title(f'{label}', fontsize=11, fontweight='bold')
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_yticks(range(matrix.shape[0]))
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Accuracy (%)', fontsize=10)
    
    # Hide unused subplots
    for idx in range(n_exps, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('Accuracy Matrices', fontsize=14, fontweight='bold', y=1.0)
    fig.tight_layout()
    
    _save_and_show(fig, out, show, 'accuracy_matrices')


def plot_training_curves(experiments: List[Tuple[str, Dict, List[Dict]]], 
                        out: Optional[Path] = None, show: bool = False):
    """Plot training loss and accuracy curves over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for label, results, epoch_logs in experiments:
        if not epoch_logs:
            continue
        
        # Group by task
        task_epochs = defaultdict(lambda: {'epochs': [], 'loss': [], 'acc': []})
        for log in epoch_logs:
            task_id = log.get('task_id', 0)
            epoch = log.get('epoch', 0)
            task_epochs[task_id]['epochs'].append(epoch)
            task_epochs[task_id]['loss'].append(log.get('train_loss', 0))
            task_epochs[task_id]['acc'].append(log.get('train_acc', 0))
        
        # Plot aggregated curves (average across tasks)
        all_epochs = []
        all_losses = []
        all_accs = []
        
        for task_id in sorted(task_epochs.keys()):
            data = task_epochs[task_id]
            all_epochs.extend(data['epochs'])
            all_losses.extend(data['loss'])
            all_accs.extend(data['acc'])
        
        if all_epochs:
            # Plot with running average
            axes[0].plot(all_epochs, all_losses, alpha=0.6, label=label)
            axes[1].plot(all_epochs, all_accs, alpha=0.6, label=label)
    
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Training Loss', fontsize=11)
    axes[0].set_title('Training Loss', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Training Accuracy', fontsize=11)
    axes[1].set_title('Training Accuracy', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    _save_and_show(fig, out, show, 'training_curves')


def plot_parameter_drift(experiments: List[Tuple[str, Dict, List[Dict]]], 
                        out: Optional[Path] = None, show: bool = False):
    """Plot parameter drift (L2 and Fisher distances) over tasks."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    has_data = False
    for label, results, param_distances in experiments:
        if not param_distances:
            continue
        
        has_data = True
        tasks = [d['task'] for d in param_distances]
        l2 = [d.get('l2_distance', 0) for d in param_distances]
        fisher_train = [d.get('fisher_distance_train', 0) for d in param_distances]
        
        axes[0].plot(tasks, l2, 'o-', label=label, linewidth=2)
        axes[1].plot(tasks, fisher_train, 's-', label=label, linewidth=2)
    
    if not has_data:
        fig.text(0.5, 0.5, 'No parameter drift data available', 
                ha='center', va='center', fontsize=12)
    else:
        axes[0].set_xlabel('After training task', fontsize=11)
        axes[0].set_ylabel('L2 Distance', fontsize=11)
        axes[0].set_title('L2 Parameter Drift', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('After training task', fontsize=11)
        axes[1].set_ylabel('Fisher Distance', fontsize=11)
        axes[1].set_title('Fisher-Weighted Parameter Drift', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    _save_and_show(fig, out, show, 'parameter_drift')


def plot_backward_transfer(experiments: List[Tuple[str, Dict[int, List[float]]]], 
                          out: Optional[Path] = None, show: bool = False):
    """Plot backward transfer for each experiment."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for label, results in experiments:
        bwt = compute_backward_transfer(results)
        if len(bwt) > 0:
            x = np.arange(1, len(bwt) + 1)
            ax.plot(x, bwt * 100, 'o-', label=label, linewidth=2)
    
    ax.set_xlabel('Task', fontsize=11)
    ax.set_ylabel('Backward Transfer (%)', fontsize=11)
    ax.set_title('Backward Transfer Over Tasks', fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    _save_and_show(fig, out, show, 'backward_transfer')


def plot_summary_statistics(experiments: List[Tuple[str, Dict[int, List[float]]]], 
                           out: Optional[Path] = None, show: bool = False):
    """Plot summary bar chart with key metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    labels = []
    avg_final_accs = []
    avg_forgetting = []
    avg_bwt = []
    
    for label, results in experiments:
        _, final_accs, forgetting = compute_summary(results)
        bwt = compute_backward_transfer(results)
        
        labels.append(label)
        avg_final_accs.append(np.nanmean(final_accs) * 100)
        avg_forgetting.append(np.nanmean(forgetting) * 100)
        avg_bwt.append(np.nanmean(bwt) * 100 if len(bwt) > 0 else 0)
    
    x = np.arange(len(labels))
    width = 0.6
    
    axes[0].bar(x, avg_final_accs, width, color='steelblue')
    axes[0].set_ylabel('Accuracy (%)', fontsize=11)
    axes[0].set_title('Average Final Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].bar(x, avg_forgetting, width, color='coral')
    axes[1].set_ylabel('Forgetting (%)', fontsize=11)
    axes[1].set_title('Average Forgetting', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)
    
    axes[2].bar(x, avg_bwt, width, color='seagreen')
    axes[2].set_ylabel('Backward Transfer (%)', fontsize=11)
    axes[2].set_title('Average Backward Transfer', fontsize=12, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45, ha='right')
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].axhline(y=0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)
    
    fig.tight_layout()
    
    _save_and_show(fig, out, show, 'summary_statistics')


def plot_all_in_one(experiments: List[Tuple[str, Dict[int, List[float]]]], 
                   out: Optional[Path] = None, show: bool = False):
    """Create the original 3-panel comparison plot."""
    n_exps = len(experiments)
    if n_exps == 0:
        raise ValueError('No experiments provided')
    
    summaries = [(label, *compute_summary(results)) for label, results in experiments]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Panel 1: mean accuracy over time
    ax = axes[0]
    for label, mean_ot, _, _ in summaries:
        x = np.arange(1, len(mean_ot) + 1)
        ax.plot(x, mean_ot, marker='o', label=label)
    ax.set_xlabel('After training task k')
    ax.set_ylabel('Mean accuracy across seen tasks')
    ax.set_title('Mean accuracy over time')
    ax.grid(True)
    ax.legend()
    
    # Panel 2: final per-task accuracies (grouped bars)
    ax = axes[1]
    max_num_tasks = max(len(s[2]) for s in summaries)
    indices = np.arange(max_num_tasks)
    width = 0.8 / max(1, n_exps)
    for i, (label, _, final_accs, _) in enumerate(summaries):
        vals = final_accs if final_accs.size else np.full(max_num_tasks, np.nan)
        if vals.size < max_num_tasks:
            vals = np.concatenate([vals, np.full(max_num_tasks - vals.size, np.nan)])
        ax.bar(indices + i * width, vals * 100.0, width=width, label=label)
    ax.set_xlabel('Task id')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Final per-task accuracies')
    ax.set_xticks(indices + width * (n_exps - 1) / 2)
    ax.set_xticklabels([str(i) for i in range(max_num_tasks)])
    ax.legend()
    ax.grid(axis='y')
    
    # Panel 3: forgetting per task (grouped bars)
    ax = axes[2]
    for i, (label, _, _, forgetting) in enumerate(summaries):
        vals = forgetting if forgetting.size else np.full(max_num_tasks, np.nan)
        if vals.size < max_num_tasks:
            vals = np.concatenate([vals, np.full(max_num_tasks - vals.size, np.nan)])
        ax.bar(indices + i * width, vals * 100.0, width=width, label=label)
    ax.set_xlabel('Task id')
    ax.set_ylabel('Forgetting (%)')
    ax.set_title('Forgetting per task')
    ax.set_xticks(indices + width * (n_exps - 1) / 2)
    ax.set_xticklabels([str(i) for i in range(max_num_tasks)])
    ax.legend()
    ax.grid(axis='y')
    
    plt.tight_layout()
    
    _save_and_show(fig, out, show, 'all_in_one')


# =============================================================================
# Utilities
# =============================================================================

def _save_and_show(fig: plt.Figure, out: Optional[Path], show: bool, name: str):
    """Helper to save and/or show a figure."""
    if out:
        if out.is_dir():
            save_path = out / f"{name}.png"
        else:
            # Use provided path but ensure proper naming for multiple plots
            save_path = out.parent / f"{out.stem}_{name}{out.suffix}"
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f'Saved {name} to {save_path}')
    
    if show:
        plt.show()
    else:
        plt.close(fig)


PLOT_FUNCTIONS = {
    'all_in_one': plot_all_in_one,
    'mean_accuracy': plot_mean_accuracy,
    'final_accuracies': plot_final_accuracies,
    'forgetting': plot_forgetting,
    'accuracy_matrices': plot_accuracy_matrices,
    'training_curves': lambda exps, out, show: plot_training_curves(
        [(l, r, extract_epoch_logs(d)) for l, r, d in exps], out, show
    ),
    'parameter_drift': lambda exps, out, show: plot_parameter_drift(
        [(l, r, extract_param_distances(d)) for l, r, d in exps], out, show
    ),
    'backward_transfer': plot_backward_transfer,
    'summary_statistics': plot_summary_statistics,
}


def main():
    p = argparse.ArgumentParser(
        description='Compare experiment results with comprehensive plotting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available plot types:
  all_in_one          - Original 3-panel comparison (default)
  mean_accuracy       - Mean accuracy over time
  final_accuracies    - Final per-task accuracies (bars)
  forgetting          - Forgetting per task (bars)
  accuracy_matrices   - Accuracy matrices as heatmaps
  training_curves     - Training loss and accuracy curves
  parameter_drift     - L2 and Fisher-weighted parameter drift
  backward_transfer   - Backward transfer over tasks
  summary_statistics  - Summary bar charts of key metrics
  all                 - Generate all available plots

Examples:
  # Original 3-panel comparison
  python compare_experiments.py exp1 exp2 --labels "OGD","FOPNG" --show
  
  # Generate specific plots
  python compare_experiments.py exp1 exp2 exp3 --plots mean_accuracy forgetting --show
  
  # Generate all plots to a directory
  python compare_experiments.py exp1 exp2 --plots all --out-dir ./comparison_plots
  
  # Single plot to file
  python compare_experiments.py exp1 exp2 --plots summary_statistics --out summary.png
        """
    )
    p.add_argument('exp_paths', nargs='+', help='Paths to experiment directories or JSON files')
    p.add_argument('--labels', help='Comma-separated labels matching exp_paths (optional)', default=None)
    p.add_argument('--out', help='Output image path (png). For single plot or all_in_one.', default=None)
    p.add_argument('--out-dir', help='Output directory for multiple plots', default=None)
    p.add_argument('--show', help='Show plot(s) interactively', action='store_true')
    p.add_argument('--plots', nargs='+', 
                   choices=list(PLOT_FUNCTIONS.keys()) + ['all'],
                   default=['all_in_one'],
                   help='Which plots to generate (default: all_in_one)')
    args = p.parse_args()
    
    # Parse labels
    labels = None
    if args.labels:
        labels = [s.strip() for s in args.labels.split(',')]
        if len(labels) != len(args.exp_paths):
            raise SystemExit('Number of labels must match number of experiment paths')
    
    # Load experiments
    experiments_with_data = []
    for i, path_str in enumerate(args.exp_paths):
        path = Path(path_str)
        data = load_experiment(path)
        try:
            results = extract_results(data)
        except Exception as e:
            raise SystemExit(f'Failed to parse experiment at {path}: {e}')
        label = labels[i] if labels else path.name
        experiments_with_data.append((label, results, data))
    
    print(f"Loaded {len(experiments_with_data)} experiments")
    
    # Prepare output path
    if args.out_dir:
        out_path = Path(args.out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
    elif args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = None
    
    # Determine which plots to generate
    plot_names = args.plots
    if 'all' in plot_names:
        plot_names = list(PLOT_FUNCTIONS.keys())
    
    # Generate plots
    experiments = [(label, results) for label, results, _ in experiments_with_data]
    
    for plot_name in plot_names:
        print(f"Generating {plot_name}...")
        
        if plot_name in ['training_curves', 'parameter_drift']:
            # These need the full data
            PLOT_FUNCTIONS[plot_name](experiments_with_data, out_path, args.show)
        else:
            PLOT_FUNCTIONS[plot_name](experiments, out_path, args.show)
    
    print("Done!")


if __name__ == '__main__':
    main()