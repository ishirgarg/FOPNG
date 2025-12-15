#!/bin/bash
seeds=(2 3 4 5)
datasets=(permuted_mnist split_mnist_ic rotated_mnist)

for seed in "${seeds[@]}"; do
  echo "Running FOPNG (alpha=$alpha, seed=$seed) on split_mnist_ic"
  python3 main.py \
      --dataset split_mnist_ic \
      --method ewc \
      --fisher diagonal \
      --num_tasks 5 \
      --epochs 5 \
      --lr 5e-4 \
      --batch_size 10 \
      --ewc_lambda 400 \
      --seed "$seed" \
      --device mps
done

for seed in "${seeds[@]}"; do
  echo "Running FOPNG (alpha=$alpha, seed=$seed) on rotated_mnist"
  python3 main.py \
      --dataset rotated_mnist \
      --method ewc \
      --fisher diagonal \
      --num_tasks 5 \
      --epochs 5 \
      --lr 1e-2 \
      --batch_size 10 \
      --ewc_lambda 50 \
      --seed "$seed" \
      --device mps
done

for seed in "${seeds[@]}"; do
  echo "Running FOPNG (alpha=$alpha, seed=$seed) on permuted_mnist"
  python3 main.py \
      --dataset permuted_mnist \
      --method ewc \
      --fisher diagonal \
      --num_tasks 5 \
      --epochs 5 \
      --lr 1e-2 \
      --batch_size 10 \
      --ewc_lambda 10 \
      --seed "$seed" \
      --device mps
done





for seed in "${seeds[@]}"; do
  echo "Running FOPNG (alpha=$alpha, seed=$seed) on split_mnist_ic"
  python3 main.py \
      --dataset split_mnist_ic \
      --method ogd \
      --num_tasks 5 \
      --epochs 5 \
      --lr 5e-4 \
      --batch_size 10 \
      --max_directions 2000 \
      --grads_per_task 80 \
      --seed "$seed" \
      --device mps
done

for seed in "${seeds[@]}"; do
  echo "Running FOPNG (alpha=$alpha, seed=$seed) on rotated_mnist"
  python3 main.py \
      --dataset rotated_mnist \
      --method ogd \
      --num_tasks 5 \
      --epochs 5 \
      --lr 5e-2 \
      --batch_size 10 \
      --max_directions 2000 \
      --grads_per_task 80 \
      --seed "$seed" \
      --device mps
done

for seed in "${seeds[@]}"; do
  echo "Running FOPNG (alpha=$alpha, seed=$seed) on permuted_mnist"
  python3 main.py \
      --dataset permuted_mnist \
      --method ogd \
      --num_tasks 5 \
      --epochs 5 \
      --lr 5e-3 \
      --batch_size 10 \
      --max_directions 2000 \
      --grads_per_task 80 \
      --seed "$seed" \
      --device mps
done
