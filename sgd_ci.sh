#!/bin/bash
seeds=(2 3 4 5)
datasets=(permuted_mnist)

for seed in "${seeds[@]}"; do
  echo "Running FOPNG (alpha=$alpha, seed=$seed) on split_mnist_ic"
  python3 main.py \
      --dataset split_mnist_ic \
      --method sgd \
      --num_tasks 5 \
      --epochs 5 \
      --lr 5e-4 \
      --batch_size 10 \
      --seed "$seed" \
      --device cpu
done
