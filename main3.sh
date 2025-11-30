#!/bin/bash

datasets=(rotated_mnist split_mnist permuted_mnist)
seeds=$(seq 1 1)

# Learning rate sweeps (6 values each)
lrs_sgd=(5e-2 1e-1)

for dataset in "${datasets[@]}"; do
  for seed in $seeds; do
    echo "==== DATASET $dataset | SEED $seed ===="

    # ---------------------- SGD ----------------------
    for lr in "${lrs_sgd[@]}"; do
      echo "Running SGD (lr=$lr, seed=$seed) on $dataset"
      python3 main.py \
          --dataset "$dataset" \
          --method sgd \
          --num_tasks 5 \
          --epochs 5 \
          --lr "$lr" \
          --batch_size 10 \
          --grads_per_task 80 \
          --seed "$seed" \
          --device mps
    done
  done
done