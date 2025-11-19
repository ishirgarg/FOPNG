#!/usr/bin/env python3
"""
compare_experiments.py

Load saved experiment outputs (JSON/PKL) produced by this repository and plot
side-by-side comparisons.

Features:
- Load experiment data from an experiment directory (tries utils.load_experiment).
- Plot mean accuracy over time (after each training task), final per-task
  accuracies (grouped bars), and forgetting per task (grouped bars).
- CLI: pass multiple experiment paths and optional labels, save or show figure.

Example:
  python scripts/compare_experiments.py \
    ./experiments/ogd_gtl_permuted ./experiments/fopng_diagonal_permuted \
    --labels "OGD-GTL","FOPNG" --out compare.png --show

"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
        # try reading file itself
        if p.exists():
            return json.loads(p.read_text())
        raise FileNotFoundError(f"No experiment data found at {p}")


def extract_results(data: Dict) -> Dict[int, List[float]]:
    """Return results mapping: task_id -> list of accuracies (evaluated after each training task).

    The loader accepts multiple experiment file formats. We try common keys.
    """
    # experiment_data.json format has top-level 'results'
    if 'results' in data and isinstance(data['results'], dict):
        return {int(k): v for k, v in data['results'].items()}

    # older format: root is results mapping already
    if all(isinstance(k, str) and isinstance(v, list) for k, v in data.items()):
        # assume keys are task ids
        try:
            return {int(k): v for k, v in data.items()}
        except ValueError:
            pass

    raise ValueError('Unrecognized experiment file format')


def compute_summary(results: Dict[int, List[float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (mean_over_time, final_accs, forgetting)

    - mean_over_time[k-1] = mean accuracy across tasks 0..k-1 evaluated after training task k-1
    - final_accs[t] = accuracy of task t after training all tasks
    - forgetting[t] = max accuracy achieved for task t during training - final_accs[t]
    """
    if not results:
        return np.array([]), np.array([]), np.array([])

    num_tasks = max(results.keys()) + 1

    # Ensure each task has a list
    task_lists = [results.get(t, []) for t in range(num_tasks)]

    # mean_over_time: for k in 1..num_tasks
    mean_over_time = []
    for k in range(1, num_tasks + 1):
        vals = []
        for t in range(k):
            lst = task_lists[t]
            if len(lst) >= k:
                vals.append(lst[k - 1])
        mean_over_time.append(np.mean(vals) if vals else np.nan)

    final_accs = np.array([task_lists[t][-1] if task_lists[t] else np.nan for t in range(num_tasks)])
    max_accs = np.array([max(task_lists[t]) if task_lists[t] else np.nan for t in range(num_tasks)])
    forgetting = max_accs - final_accs

    return np.array(mean_over_time), final_accs, forgetting


def plot_comparison(experiments: List[Tuple[str, Dict[int, List[float]]]], out: Optional[Path] = None, show: bool = False):
    """Make a 1x3 figure comparing experiments.

    experiments: list of (label, results_dict)
    """
    n_exps = len(experiments)
    if n_exps == 0:
        raise ValueError('No experiments provided')

    # Determine max tasks across experiments for consistent x-axis
    max_tasks = 0
    summaries = []  # (label, mean_over_time, final_accs, forgetting)
    for label, res in experiments:
        mean_ot, final_accs, forgetting = compute_summary(res)
        summaries.append((label, mean_ot, final_accs, forgetting))
        max_tasks = max(max_tasks, len(mean_ot))

    # Create figure
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
    # Determine max number of tasks among experiments
    max_num_tasks = max(len(s[2]) for s in summaries)
    indices = np.arange(max_num_tasks)
    width = 0.8 / max(1, n_exps)
    for i, (label, _, final_accs, _) in enumerate(summaries):
        vals = final_accs if final_accs.size else np.full(max_num_tasks, np.nan)
        # pad if needed
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
    if out:
        fig.savefig(out, dpi=200)
        print(f'Saved figure to {out}')
    if show:
        plt.show()
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description='Compare experiment results and plot side-by-side')
    p.add_argument('exp_paths', nargs='+', help='Paths to experiment directories or JSON files')
    p.add_argument('--labels', help='Comma-separated labels matching exp_paths (optional)', default=None)
    p.add_argument('--out', help='Output image path (png). If omitted, figure is not saved.', default=None)
    p.add_argument('--show', help='Show plot interactively', action='store_true')
    args = p.parse_args()

    labels = None
    if args.labels:
        labels = [s.strip() for s in args.labels.split(',')]
        if len(labels) != len(args.exp_paths):
            raise SystemExit('Number of labels must match number of experiment paths')

    experiments = []
    for i, path_str in enumerate(args.exp_paths):
        path = Path(path_str)
        data = load_experiment(path)
        try:
            res = extract_results(data)
        except Exception as e:
            raise SystemExit(f'Failed to parse experiment at {path}: {e}')
        label = labels[i] if labels else path.name
        experiments.append((label, res))

    out = Path(args.out) if args.out else None
    plot_comparison(experiments, out=out, show=args.show)


if __name__ == '__main__':
    main()
