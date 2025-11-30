#!/bin/bash

datasets=(rotated_mnist split_mnist permuted_mnist)
seeds=$(seq 1 1)

lrs_fng=(5e-4 1e-3 5e-3 1e-2)

for dataset in "${datasets[@]}"; do
  for seed in $seeds; do

    echo "==== DATASET $dataset | SEED $seed ===="

    for lr in "${lrs_fng[@]}"; do
      echo "Running FNG (lr=$lr, seed=$seed) on $dataset"
      python3 main.py \
          --dataset "$dataset" \
          --method fng \
          --fisher diagonal \
          --num_tasks 5 \
          --epochs 5 \
          --lr "$lr" \
          --grads_per_task 80 \
          --fopng_lambda_reg 1e-3 \
          --batch_size 10 \
          --seed "$seed" \
          --device cuda 
    done

  done
done
