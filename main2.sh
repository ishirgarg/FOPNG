#!/bin/bash

datasets=(permuted_mnist)
seeds=$(seq 1 1)

# Learning rate sweeps (6 values each)
lambda_fopng=(1e-2 7e-3 5e-3 3e-3 7e-4 5e-4 3e-4 1e-4 7e-5 5e-5 3e-5 1e-5)

for dataset in "${datasets[@]}"; do
  for seed in $seeds; do

    echo "==== DATASET $dataset | SEED $seed ===="

    # ---------------------- FOPNG ----------------------
    for lambda in "${lambda_fopng[@]}"; do
      echo "Running FOPNG (lambda=$lambda, seed=$seed) on $dataset"
      python3 main.py \
          --dataset "$dataset" \
          --method fopng \
          --fisher diagonal \
          --num_tasks 5 \
          --epochs 5 \
          --lr 1e-4 \
          --collector gtl \
          --max_directions 2000 \
          --grads_per_task 80 \
          --fopng_lambda_reg "$lambda" \
          --fopng_new_fisher_weight 0.5 \
          --batch_size 10 \
          --seed "$seed" \
          --device mps
    done
  done
done
