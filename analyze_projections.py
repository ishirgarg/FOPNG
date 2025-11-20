#!/usr/bin/env python3
"""Analyze projection ratios from the comparison log."""

# From the log file, I extracted:
# LOW MOMENTUM (0.3):
# Task 1, Epoch 1: 0.201%
# Task 1, Epoch 2: 0.054%
# Task 2, Epoch 1: 0.357%
# Task 2, Epoch 2: 0.090%

# HIGH MOMENTUM (0.8):
# Task 1, Epoch 1: 0.201%
# Task 1, Epoch 2: 0.054%
# Task 2, Epoch 1: 0.281%
# Task 2, Epoch 2: 0.080%

print("\n" + "="*80)
print("PROJECTION RATIO ANALYSIS")
print("="*80)
print("\nFormat: Task T, Epoch E | Momentum 0.3 | Momentum 0.8 | Δ Absolute | Δ Relative")
print("-"*80)

data = [
    (1, 1, 0.201, 0.201),
    (1, 2, 0.054, 0.054),
    (2, 1, 0.357, 0.281),
    (2, 2, 0.090, 0.080),
]

for task, epoch, low, high in data:
    diff_abs = abs(low - high)
    diff_rel = (diff_abs / low * 100) if low > 0 else 0
    match = "✓ IDENTICAL" if diff_abs < 0.001 else "✗ DIFFERENT"
    
    print(f"Task {task}, Epoch {epoch}: {low:6.3f}% | {high:6.3f}% | "
          f"Δ={diff_abs:6.3f}% ({diff_rel:+6.1f}% rel) {match}")

print("\n" + "="*80)
print("FINAL ACCURACY COMPARISON")
print("="*80)

# From the log
acc_data = [
    (0, 91.37, 91.37),
    (1, 81.89, 81.88),
    (2, 85.17, 85.19),
]

for task, low, high in acc_data:
    diff = abs(low - high)
    print(f"Task {task}: {low:.2f}% | {high:.2f}% | Δ={diff:.4f}%")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)
print("\n1. Task 1 projections: IDENTICAL (0% difference)")
print("   → F_old too small at this point, barely contributes")
print("\n2. Task 2 projections: DIFFERENT (21-27% relative difference)")
print("   → Momentum 0.3: 0.357% and 0.090%")
print("   → Momentum 0.8: 0.281% and 0.080%")
print("   → Projection ratios differ by up to 0.076 percentage points")
print("\n3. But final accuracies: NEARLY IDENTICAL (max 0.02% difference)")
print("   → Despite 21% different projection ratios!")
print("\n" + "="*80)
print("CONCLUSION: Projections ARE different, but effects cancel out!")
print("="*80)
print("\nWhy? Possible explanations:")
print("  - Projections are still tiny (< 0.4% of gradient)")
print("  - Over ~6000 updates/task, random variations dominate")
print("  - The 0.076% difference in correction is swamped by 99.6% gradient")
print("  - Both converge to similar local minima despite different paths")
print("="*80 + "\n")
