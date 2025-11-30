#!/bin/bash

datasets=(rotated_mnist split_mnist permuted_mnist)
seeds=$(seq 1 1)

# Learning rate sweeps (6 values each)
lrs_fopng=(5e-3 1e-2)

for dataset in "${datasets[@]}"; do
  for seed in $seeds; do
    echo "==== DATASET $dataset | SEED $seed ===="

    # ---------------------- FOPNG ----------------------
    for lr in "${lrs_fopng[@]}"; do
      echo "Running FOPNG (lr=$lr, seed=$seed) on $dataset"
      python3 main.py \
          --dataset "$dataset" \
          --method fopng \
          --fisher diagonal \
          --num_tasks 5 \
          --epochs 5 \
          --lr "$lr" \
          --collector gtl \
          --max_directions 2000 \
          --grads_per_task 80 \
          --fopng_lambda_reg 1e-3 \
          --fopng_new_fisher_weight 0.5 \
          --batch_size 10 \
          --seed "$seed" \
          --device mps
    done
  done
done