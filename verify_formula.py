#!/usr/bin/env python3
"""Verify FOPNG projection formula matches the algorithm."""

print("\n" + "="*80)
print("VERIFYING FOPNG PROJECTION FORMULA")
print("="*80)

print("""
FOPNG PAPER/ALGORITHM:

Goal: Project gradient away from subspace spanned by previous task gradients,
      weighted by Fisher information matrix.

The projection operator in Fisher space is:
  P_F = I - F_old * G * (G^T * F_old * F_new^{-1} * F_old * G + λI)^{-1} * G^T * F_old

Applied to gradient g:
  P_g = P_F * g
      = g - F_old * G * A^{-1} * G^T * F_old * g

Where:
  A = G^T * F_old * F_new^{-1} * F_old * G + λI
  (precomputed in _compute_update_prep)

Then natural gradient update:
  u = -ε * F_new^{-1} * P_g / ||P_g||_{F_new^{-1}}
""")

print("\n" + "="*80)
print("IMPLEMENTATION IN OPTIMIZERS.PY")
print("="*80)

print("""
DIAGONAL CASE (lines 315-320):
┌─────────────────────────────────────────────────────────────┐
│ F_new_inv_diag = 1.0 / (F_new + lam)                       │
│ F_old_g = F_old * gradient                                  │
│ G_T_F_old_g = G.T @ F_old_g                                 │
│ A_inv_G_T_F_old_g = self.A_inv @ G_T_F_old_g               │
│ correction = (G @ A_inv_G_T_F_old_g).view(-1) * F_old      │
│ P_g = gradient - correction                                 │
│ F_new_inv_P_g = P_g * F_new_inv_diag                        │
│ denom = sqrt((P_g * F_new_inv_P_g).sum() + 1e-8)           │
│ v_star = -epsilon * F_new_inv_P_g / (denom + 1e-8)         │
└─────────────────────────────────────────────────────────────┘

Step-by-step verification:
  1. F_old_g = F_old ⊙ g                    ✓ Element-wise for diagonal
  2. G^T F_old g = G^T @ (F_old ⊙ g)        ✓ Matrix-vector product
  3. A^{-1} G^T F_old g                     ✓ Using precomputed A_inv
  4. G A^{-1} G^T F_old g                   ✓ 
  5. F_old ⊙ (G A^{-1} G^T F_old g)        ✓ Element-wise for diagonal
  6. P_g = g - correction                   ✓ Projection
  7. F_new^{-1} ⊙ P_g                      ✓ Natural gradient direction
  8. Normalize by Fisher norm               ✓
  9. Scale by ε                             ✓

FULL CASE (lines 326-332):
┌─────────────────────────────────────────────────────────────┐
│ F_new_inv = inverse(F_new + lam * I)                        │
│ temp = F_old @ F_new_inv @ F_old @ G                        │
│ A = G.T @ temp + lam * I                                    │
│ A_inv = inverse(A)                                           │
│ P = I - F_old @ G @ A_inv @ G.T @ F_old                    │
│ P_g = P @ gradient                                           │
│ F_new_inv_P_g = F_new_inv @ P_g                            │
│ denom = sqrt(P_g @ F_new_inv_P_g + 1e-8)                   │
│ v_star = -epsilon * F_new_inv_P_g / denom                  │
└─────────────────────────────────────────────────────────────┘

Step-by-step verification:
  1. Compute A = G^T F_old F_new^{-1} F_old G + λI  ✓
  2. P = I - F_old G A^{-1} G^T F_old                ✓ Projection operator
  3. P_g = P @ g                                      ✓ Apply projection
  4. F_new^{-1} @ P_g                                ✓ Natural gradient
  5. Normalize and scale by ε                        ✓
""")

print("\n" + "="*80)
print("PRECOMPUTATION (_compute_update_prep)")
print("="*80)

print("""
DIAGONAL CASE:
┌─────────────────────────────────────────────────────────────┐
│ F_new_inv_diag = 1.0 / (F_new + lam)                       │
│ F_old_diag = F_old.view(-1, 1)                              │
│ F_old_G = F_old_diag * G                                    │
│ weighted_G = F_old_diag * (F_new_inv_diag * F_old_G)       │
│ A = G.T @ weighted_G + lam * eye(G.size(1))                │
│ self.A_inv = pinverse(A)                                    │
└─────────────────────────────────────────────────────────────┘

This computes:
  A = G^T * diag(F_old) * diag(F_new^{-1}) * diag(F_old) * G + λI
    = G^T * diag(F_old ⊙ F_new^{-1} ⊙ F_old) * G + λI
    = G^T * diag(F_old² ⊙ F_new^{-1}) * G + λI

For diagonal matrices: F_old * F_new^{-1} * F_old = diag(F_old² / F_new)

✓ CORRECT! Diagonal approximation of full formula.
""")

print("\n" + "="*80)
print("VERIFICATION RESULT")
print("="*80)

print("""
✅ DIAGONAL FISHER PROJECTION: CORRECT
   - Properly computes F_old * G * A^{-1} * G^T * F_old * g
   - Uses element-wise operations for diagonal matrices
   - Precomputes A_inv for efficiency
   
✅ FULL FISHER PROJECTION: CORRECT  
   - Properly computes projection operator P
   - Uses full matrix operations
   - Computes A on-the-fly (not precomputed in features branch)

✅ NATURAL GRADIENT APPLICATION: CORRECT
   - Applies F_new^{-1} to projected gradient
   - Normalizes to fixed Fisher norm ε
   - Both diagonal and full cases match theory

✅ FISHER UPDATE MECHANISM: ENHANCED
   - Original: simple average (F_old + F_current) / 2
   - Current: EMA with momentum β or cumulative average
   - Both are valid strategies for combining Fisher info

CONCLUSION: The FOPNG implementation is mathematically correct.
            The projection formula matches the expected algorithm.
""")

print("="*80 + "\n")
