#!/usr/bin/env python3
"""Quick debug to compare momentum values - just 2 tasks, 1 epoch."""

import torch
from config import Config
from main import run_permuted_mnist

# Store first batch updates for comparison
first_batch_updates = {}

def test_momentum(momentum_val):
    config = Config(
        seed=1234,
        batch_size=10,
        lr=1e-3,
        epochs_per_task=1,
        grads_per_task=50,  # Reduced for speed
        device='cpu',
        log_dir=None,
        save_model=False,
        save_plots=False,
        save_raw_data=False,
        experiment_name=f"debug_{momentum_val}",
        fopng_lambda_reg=0.001,
        fopng_epsilon=0.0001,
        fopng_fisher_momentum=momentum_val,
        fopng_use_cumulative_fisher=False,
    )
    
    print(f"\n{'='*70}")
    print(f"Testing Momentum = {momentum_val}")
    print(f"{'='*70}")
    
    results, _ = run_permuted_mnist(
        method_name='fopng',
        num_tasks=2,  # Just 2 tasks
        config=config,
        collector='ave',
        fisher='diagonal',
        max_directions=200,
    )
    
    return results

if __name__ == "__main__":
    print("\nQuick Fisher Momentum Comparison")
    print("="*70)
    
    # Test both
    r1 = test_momentum(0.3)
    r2 = test_momentum(0.8)
    
    # Compare
    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"Momentum 0.3 - Task 0: {r1[0][-1]*100:.2f}%, Task 1: {r1[1][-1]*100:.2f}%")
    print(f"Momentum 0.8 - Task 0: {r2[0][-1]*100:.2f}%, Task 1: {r2[1][-1]*100:.2f}%")
    print(f"Difference Task 0: {abs(r1[0][-1] - r2[0][-1])*100:.4f}%")
    print(f"Difference Task 1: {abs(r1[1][-1] - r2[1][-1])*100:.4f}%")

