#!/usr/bin/env python3
"""
Compare projection ratios between different Fisher momentum values.
"""

import sys
import torch
from config import Config
from main import run_permuted_mnist

def run_with_momentum(momentum_val, label):
    """Run experiment with specific momentum and return results."""
    print(f"\n{'='*80}")
    print(f"{'='*80}")
    print(f"EXPERIMENT: {label} (Fisher momentum = {momentum_val})")
    print(f"{'='*80}")
    print(f"{'='*80}\n")
    
    config = Config(
        seed=1234,
        batch_size=10,
        lr=1e-3,
        epochs_per_task=2,  # 2 epochs for speed
        grads_per_task=100,  # Reduced for speed
        device='cpu',
        log_dir=None,
        save_model=False,
        save_plots=False,
        save_raw_data=True,  # Enable stats collection!
        experiment_name=f"compare_momentum_{momentum_val}",
        fopng_lambda_reg=0.001,
        fopng_epsilon=0.0001,
        fopng_fisher_momentum=momentum_val,
        fopng_use_cumulative_fisher=False,
    )
    
    results, logger = run_permuted_mnist(
        method_name='fopng',
        num_tasks=3,
        config=config,
        collector='ave',
        fisher='diagonal',
        max_directions=500,
    )
    
    # Extract projection ratios from logger if available
    projection_data = {}
    if logger and logger.epoch_logs:
        for log in logger.epoch_logs:
            key = (log.task_id, log.epoch)
            projection_data[key] = {
                'projection_ratio': log.projection_ratio_mean,
                'correction_norm': log.correction_norm_mean,
                'grad_norm': log.grad_norm_mean,
                'update_norm': log.update_norm_mean,
            }
    
    return results, projection_data

if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# PROJECTION RATIO COMPARISON")
    print("# Testing: Does Fisher momentum change projection ratios?")
    print("#"*80)
    
    # Run both experiments
    results_low, proj_low = run_with_momentum(0.3, "LOW MOMENTUM")
    results_high, proj_high = run_with_momentum(0.8, "HIGH MOMENTUM")
    
    # Compare projection ratios
    print("\n" + "="*80)
    print("PROJECTION RATIO COMPARISON")
    print("="*80)
    print("\nFormat: Task T, Epoch E | Momentum 0.3 | Momentum 0.8 | Difference")
    print("-"*80)
    
    all_keys = sorted(set(proj_low.keys()) | set(proj_high.keys()))
    
    for key in all_keys:
        task_id, epoch = key
        if key in proj_low and key in proj_high:
            low_val = proj_low[key]['projection_ratio']
            high_val = proj_high[key]['projection_ratio']
            
            if low_val is not None and high_val is not None:
                diff = abs(low_val - high_val)
                rel_diff = diff / (low_val + 1e-10) * 100
                
                print(f"Task {task_id}, Epoch {epoch}: "
                      f"{low_val*100:6.3f}% | {high_val*100:6.3f}% | "
                      f"Δ={diff*100:6.3f}% ({rel_diff:+6.1f}% relative)")
    
    # Compare final accuracies
    print("\n" + "="*80)
    print("FINAL ACCURACY COMPARISON")
    print("="*80)
    
    for task_id in sorted(results_low.keys()):
        acc_low = results_low[task_id][-1]
        acc_high = results_high[task_id][-1]
        diff = abs(acc_low - acc_high)
        print(f"Task {task_id}: "
              f"Low={acc_low*100:.2f}% | High={acc_high*100:.2f}% | "
              f"Δ={diff*100:.4f}%")
    
    print("\n" + "="*80)
    print("KEY QUESTION: Are projection ratios different but accuracies same?")
    print("="*80 + "\n")

