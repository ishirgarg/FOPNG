#!/usr/bin/env python3
"""
filter_experiments.py

Search through experiment directories, filter by config parameters,
and generate comparison plots for matching experiments.

Features:
- Recursively search a base directory for experiment data
- Filter experiments by arbitrary config parameters
- Generate comparison plots for all matching experiments
- Support multiple filter criteria with AND/OR logic

Example usage:
  # Find all OGD experiments with lr=0.01
  python filter_experiments.py ./experiments --filter method=ogd lr=0.01
  
  # Find all experiments on split_mnist dataset
  python filter_experiments.py ./experiments --filter dataset=split_mnist --out comparison.png
  
  # Compare different learning rates for the same method
  python filter_experiments.py ./experiments --filter method=fopng dataset=split_mnist --group-by lr
  
  # Show only experiments with specific Fisher type
  python filter_experiments.py ./experiments --filter fisher_type=diagonal --show
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


def find_experiments(base_path: Path, recursive: bool = True) -> List[Path]:
    """Find all experiment directories under base_path."""
    experiments = []
    
    if recursive:
        # Look for experiment_data.json or results.json files
        for json_file in base_path.rglob("experiment_data.json"):
            experiments.append(json_file.parent)
        for json_file in base_path.rglob("results.json"):
            parent = json_file.parent
            # Avoid duplicates if both files exist
            if parent not in experiments:
                experiments.append(parent)
    else:
        # Only look in immediate subdirectories
        for subdir in base_path.iterdir():
            if subdir.is_dir():
                if (subdir / "experiment_data.json").exists() or (subdir / "results.json").exists():
                    experiments.append(subdir)
    
    return sorted(experiments)


def get_config_value(data: Dict, key: str) -> Any:
    """Extract a config value from experiment data."""
    # Try metadata.config first
    if 'metadata' in data and 'config' in data['metadata']:
        config = data['metadata']['config']
        if key in config:
            return config[key]
    
    # Try direct config key
    if 'config' in data and key in data['config']:
        return data['config'][key]
    
    # Try metadata for common fields
    if 'metadata' in data:
        if key == 'method' or key == 'method_name':
            return data['metadata'].get('method_name')
        if key == 'dataset' or key == 'dataset_name':
            return data['metadata'].get('dataset_name')
    
    return None


def matches_filters(data: Dict, filters: Dict[str, Any]) -> bool:
    """Check if experiment data matches all filter criteria."""
    for key, expected_value in filters.items():
        actual_value = get_config_value(data, key)
        
        # Handle None cases
        if actual_value is None:
            return False
        
        # Convert to strings for comparison to handle different types
        if str(actual_value).lower() != str(expected_value).lower():
            return False
    
    return True


def extract_results(data: Dict) -> Dict[int, List[float]]:
    """Extract results from experiment data."""
    if 'results' in data and isinstance(data['results'], dict):
        # Handle both string and int keys
        results = {}
        for k, v in data['results'].items():
            try:
                results[int(k)] = v if isinstance(v, list) else [v]
            except (ValueError, TypeError):
                pass
        return results
    
    # Try direct format
    try:
        return {int(k): v for k, v in data.items() if isinstance(v, list)}
    except (ValueError, TypeError):
        return {}


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


def create_experiment_label(data: Dict, group_by: Optional[str] = None) -> str:
    """Create a descriptive label for an experiment."""
    method = get_config_value(data, 'method_name') or 'Unknown'
    dataset = get_config_value(data, 'dataset_name') or 'Unknown'
    
    if group_by:
        group_value = get_config_value(data, group_by)
        if group_value is not None:
            return f"{method} ({group_by}={group_value})"
    
    # Try to get experiment name
    if 'metadata' in data:
        exp_name = data['metadata'].get('experiment_name', '')
        if exp_name and exp_name != 'noname':
            return f"{method} - {exp_name}"
    
    return f"{method} ({dataset})"


def plot_comparison(
    experiments: List[Tuple[str, Path, Dict[int, List[float]]]],
    out: Optional[Path] = None,
    show: bool = False,
    title_suffix: str = ""
):
    """
    Create comprehensive comparison plots.
    
    Args:
        experiments: List of (label, path, results) tuples
        out: Output path for saving
        show: Whether to display interactively
        title_suffix: Additional text for plot titles
    """
    if not experiments:
        print("No experiments to plot")
        return
    
    n_exps = len(experiments)
    summaries = []
    
    for label, path, results in experiments:
        mean_ot, final_accs, forgetting = compute_summary(results)
        summaries.append((label, mean_ot, final_accs, forgetting))
    
    # Determine layout based on number of experiments
    if n_exps <= 5:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        # We'll use 3 subplots and leave one for a legend or summary
    
    # Panel 1: Mean accuracy over time
    ax = axes[0] if n_exps <= 5 else axes[0, 0]
    for label, mean_ot, _, _ in summaries:
        x = np.arange(1, len(mean_ot) + 1)
        ax.plot(x, mean_ot, marker='o', label=label)
    ax.set_xlabel('After training task k')
    ax.set_ylabel('Mean accuracy across seen tasks')
    ax.set_title(f'Mean Accuracy Over Time{title_suffix}')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize='small')
    
    # Panel 2: Final per-task accuracies
    ax = axes[1] if n_exps <= 5 else axes[0, 1]
    max_num_tasks = max(len(s[2]) for s in summaries)
    indices = np.arange(max_num_tasks)
    width = 0.8 / max(1, n_exps)
    
    for i, (label, _, final_accs, _) in enumerate(summaries):
        vals = final_accs if final_accs.size else np.full(max_num_tasks, np.nan)
        if vals.size < max_num_tasks:
            vals = np.concatenate([vals, np.full(max_num_tasks - vals.size, np.nan)])
        ax.bar(indices + i * width, vals * 100.0, width=width, label=label)
    
    ax.set_xlabel('Task ID')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Final Per-Task Accuracies{title_suffix}')
    ax.set_xticks(indices + width * (n_exps - 1) / 2)
    ax.set_xticklabels([str(i) for i in range(max_num_tasks)])
    ax.legend(fontsize='small')
    ax.grid(axis='y', alpha=0.3)
    
    # Panel 3: Forgetting per task
    ax = axes[2] if n_exps <= 5 else axes[1, 0]
    for i, (label, _, _, forgetting) in enumerate(summaries):
        vals = forgetting if forgetting.size else np.full(max_num_tasks, np.nan)
        if vals.size < max_num_tasks:
            vals = np.concatenate([vals, np.full(max_num_tasks - vals.size, np.nan)])
        ax.bar(indices + i * width, vals * 100.0, width=width, label=label)
    
    ax.set_xlabel('Task ID')
    ax.set_ylabel('Forgetting (%)')
    ax.set_title(f'Forgetting Per Task{title_suffix}')
    ax.set_xticks(indices + width * (n_exps - 1) / 2)
    ax.set_xticklabels([str(i) for i in range(max_num_tasks)])
    ax.legend(fontsize='small')
    ax.grid(axis='y', alpha=0.3)
    
    # Panel 4: Summary statistics (if 2x2 layout)
    if n_exps > 5:
        ax = axes[1, 1]
        ax.axis('off')
        
        # Create summary table
        summary_text = "Summary Statistics\n" + "="*40 + "\n\n"
        for label, mean_ot, final_accs, forgetting in summaries:
            avg_final = np.nanmean(final_accs) * 100
            avg_forgetting = np.nanmean(forgetting) * 100
            summary_text += f"{label}:\n"
            summary_text += f"  Avg Final Acc: {avg_final:.2f}%\n"
            summary_text += f"  Avg Forgetting: {avg_forgetting:.2f}%\n\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if out:
        fig.savefig(out, dpi=200, bbox_inches='tight')
        print(f'Saved comparison plot to {out}')
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def print_experiment_summary(experiments: List[Tuple[str, Path, Dict]]):
    """Print a summary of found experiments."""
    print(f"\nFound {len(experiments)} matching experiments:\n")
    print(f"{'#':<4} {'Label':<40} {'Path':<50}")
    print("=" * 95)
    
    for i, (label, path, data) in enumerate(experiments, 1):
        # Truncate label and path if too long
        label_short = label[:37] + "..." if len(label) > 40 else label
        path_short = str(path.relative_to(path.parents[2]) if len(path.parts) > 2 else path)
        path_short = path_short[:47] + "..." if len(path_short) > 50 else path_short
        
        print(f"{i:<4} {label_short:<40} {path_short:<50}")
    
    print()


def group_experiments(
    experiments: List[Tuple[str, Path, Dict]],
    group_by: str
) -> Dict[Any, List[Tuple[str, Path, Dict]]]:
    """Group experiments by a config parameter."""
    groups = defaultdict(list)
    
    for label, path, data in experiments:
        group_value = get_config_value(data, group_by)
        if group_value is not None:
            groups[group_value].append((label, path, data))
    
    return dict(groups)


def main():
    parser = argparse.ArgumentParser(
        description='Filter and compare experiments by config parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find all experiments with specific method
  %(prog)s ./experiments --filter method_name=ogd
  
  # Filter by multiple parameters
  %(prog)s ./experiments --filter dataset_name=split_mnist lr=0.01
  
  # Group results by a parameter
  %(prog)s ./experiments --filter method_name=fopng --group-by lr
  
  # Save and show comparison
  %(prog)s ./experiments --filter dataset_name=permuted_mnist --out comparison.png --show
        """
    )
    
    parser.add_argument('base_path', type=str, help='Base directory to search for experiments')
    parser.add_argument('--filter', nargs='+', metavar='KEY=VALUE',
                       help='Filter criteria as key=value pairs (e.g., method=ogd lr=0.01)')
    parser.add_argument('--group-by', type=str, metavar='PARAM',
                       help='Group experiments by this config parameter and create separate plots')
    parser.add_argument('--out', type=str, help='Output path for comparison plot(s)')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    parser.add_argument('--no-recursive', action='store_true',
                       help='Only search immediate subdirectories')
    parser.add_argument('--list-only', action='store_true',
                       help='Only list matching experiments without plotting')
    parser.add_argument('--summary', action='store_true',
                       help='Print summary statistics for matching experiments')
    
    args = parser.parse_args()
    
    base_path = Path(args.base_path)
    if not base_path.exists():
        print(f"Error: Base path '{base_path}' does not exist")
        return 1
    
    # Parse filter criteria
    filters = {}
    if args.filter:
        for item in args.filter:
            if '=' not in item:
                print(f"Warning: Ignoring invalid filter '{item}' (expected KEY=VALUE)")
                continue
            key, value = item.split('=', 1)
            filters[key.strip()] = value.strip()
    
    # Find all experiments
    print(f"Searching for experiments in {base_path}...")
    all_experiments = find_experiments(base_path, recursive=not args.no_recursive)
    print(f"Found {len(all_experiments)} total experiments")
    
    # Load and filter experiments
    matching = []
    for exp_path in all_experiments:
        try:
            data = load_experiment(exp_path)
            if filters and not matches_filters(data, filters):
                continue
            
            results = extract_results(data)
            if not results:
                print(f"Warning: No results found in {exp_path}")
                continue
            
            label = create_experiment_label(data, args.group_by)
            matching.append((label, exp_path, results))
        
        except Exception as e:
            print(f"Warning: Failed to load {exp_path}: {e}")
            continue
    
    if not matching:
        print("\nNo matching experiments found!")
        if filters:
            print("Filters applied:")
            for k, v in filters.items():
                print(f"  {k} = {v}")
        return 1
    
    # Print summary of found experiments
    print_experiment_summary([(label, path, None) for label, path, _ in matching])
    
    # Print statistics if requested
    if args.summary:
        print("\nSummary Statistics:")
        print("=" * 80)
        for label, path, results in matching:
            _, final_accs, forgetting = compute_summary(results)
            avg_acc = np.nanmean(final_accs) * 100
            avg_forget = np.nanmean(forgetting) * 100
            print(f"\n{label}:")
            print(f"  Average Final Accuracy: {avg_acc:.2f}%")
            print(f"  Average Forgetting: {avg_forget:.2f}%")
            print(f"  Path: {path}")
        print()
    
    if args.list_only:
        return 0
    
    # Group and plot
    if args.group_by:
        # Load full data for grouping
        matching_with_data = []
        for label, path, results in matching:
            data = load_experiment(path)
            matching_with_data.append((label, path, data))
        
        groups = group_experiments(matching_with_data, args.group_by)
        
        print(f"\nGrouping by '{args.group_by}':")
        for group_value, group_exps in groups.items():
            print(f"  {group_value}: {len(group_exps)} experiments")
        print()
        
        # Create separate plots for each group
        for group_value, group_exps in groups.items():
            group_results = [(label, path, extract_results(data)) 
                           for label, path, data in group_exps]
            
            out_path = None
            if args.out:
                out_base = Path(args.out)
                out_path = out_base.parent / f"{out_base.stem}_{group_value}{out_base.suffix}"
            
            title_suffix = f" ({args.group_by}={group_value})"
            plot_comparison(group_results, out=out_path, show=args.show, 
                          title_suffix=title_suffix)
    else:
        # Single comparison plot
        out_path = Path(args.out) if args.out else None
        plot_comparison(matching, out=out_path, show=args.show)
    
    return 0


if __name__ == '__main__':
    exit(main())