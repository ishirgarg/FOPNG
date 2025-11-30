#!/bin/bash

datasets=(rotated_mnist split_mnist permuted_mnist)
seeds=$(seq 2 5)

for dataset in "${datasets[@]}"; do
  for seed in $seeds; do

    echo "==== DATASET $dataset | SEED $seed ===="

    # ---------------------- SGD ----------------------
    echo "Running SGD (lr=$lr, seed=$seed) on $dataset"
    python3 main.py \
        --dataset "$dataset" \
        --method sgd \
        --num_tasks 5 \
        --grads_per_task 80 \
        --epochs 5 \
        --lr 0.05 \
        --batch_size 10 \
        --seed "$seed" \
        --device cpu
  done
done
