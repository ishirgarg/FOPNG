#!/usr/bin/env python3
"""
Quick debug script to compare different Fisher momentum values.
Runs 2 tasks with 1 epoch each to see update differences.
"""

import sys
import torch
from config import Config
from main import run_permuted_mnist

def run_debug_experiment(momentum_value, label):
    """Run a quick experiment with given momentum."""
    print("\n" + "="*80)
    print(f"DEBUG RUN: {label} (momentum={momentum_value})")
    print("="*80 + "\n")
    
    config = Config(
        seed=1234,  # Same seed for reproducibility
        batch_size=10,
        lr=1e-3,
        epochs_per_task=1,  # Just 1 epoch for quick debug
        grads_per_task=200,
        device='cpu',  # Use CPU for consistency
        log_dir=None,  # Don't save
        save_model=False,
        save_plots=False,
        save_raw_data=False,
        experiment_name=f"debug_momentum_{momentum_value}",
        fopng_lambda_reg=0.001,
        fopng_epsilon=0.0001,
        fopng_fisher_momentum=momentum_value,
        fopng_use_cumulative_fisher=False,
    )
    
    results, _ = run_permuted_mnist(
        method_name='fopng',
        num_tasks=3,  # Just 3 tasks for quick test
        config=config,
        collector='ave',
        fisher='diagonal',
        max_directions=2000,
        use_cumulative_fisher=False,
    )
    
    return results

if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# FISHER MOMENTUM DEBUG COMPARISON")
    print("# Testing why different momentum values give same results")
    print("#"*80)
    
    # Run with momentum 0.3 (favor new tasks)
    results_low = run_debug_experiment(0.3, "LOW MOMENTUM (favor recent tasks)")
    
    print("\n" + "="*80)
    print("SWITCHING TO HIGH MOMENTUM...")
    print("="*80)
    
    # Run with momentum 0.8 (favor old tasks)
    results_high = run_debug_experiment(0.8, "HIGH MOMENTUM (favor old tasks)")
    
    # Compare results
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    print("\nFinal Accuracies (Low Momentum = 0.3):")
    for task_id in sorted(results_low.keys()):
        acc = results_low[task_id][-1]
        print(f"  Task {task_id}: {acc*100:.2f}%")
    
    print("\nFinal Accuracies (High Momentum = 0.8):")
    for task_id in sorted(results_high.keys()):
        acc = results_high[task_id][-1]
        print(f"  Task {task_id}: {acc*100:.2f}%")
    
    print("\nDifferences:")
    for task_id in sorted(results_low.keys()):
        diff = abs(results_low[task_id][-1] - results_high[task_id][-1])
        print(f"  Task {task_id}: {diff*100:.4f}%")
    
    print("\n" + "="*80)
    print("Check the '[FIRST BATCH DEBUG]' sections above to see:")
    print("  1. Are corrections different?")
    print("  2. Are update directions different?")
    print("  3. Do they converge to same place despite differences?")
    print("="*80 + "\n")

