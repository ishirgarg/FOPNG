#!/bin/bash

datasets=(rotated_mnist split_mnist permuted_mnist)
seeds=$(seq 1 1)

# Learning rate sweeps (6 values each)
lrs_adam=(1e-3 5e-3 1e-2 5e-3 1e-2 5e-2)

for dataset in "${datasets[@]}"; do
  for seed in $seeds; do

    echo "==== DATASET $dataset | SEED $seed ===="

    for lr in "${lrs_adam[@]}"; do
      echo "Running Adam (lr=$lr, seed=$seed) on $dataset"
      python3 main.py \
          --dataset "$dataset" \
          --method adam \
          --grads_per_task 80 \
          --num_tasks 5 \
          --epochs 5 \
          --lr "$lr" \
          --batch_size 10 \
          --seed "$seed" \
          --device cpu
    done
  done
done
