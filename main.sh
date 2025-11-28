#!/bin/bash

for seed in {1..5}
do
  echo "Running OGD seed $seed"
  python3 main.py \
      --dataset rotated_mnist \
      --method ogd \
      --num_tasks 5 \
      --epochs 5 \
      --lr 1e-3 \
      --collector gtl \
      --max_directions 200 \
      --batch_size 10 \
      --seed $seed \
      --device mps

  echo "Running FOPNG seed $seed"
  python3 main.py \
      --dataset rotated_mnist \
      --method fopng \
      --fisher diagonal \
      --num_tasks 5 \
      --epochs 5 \
      --lr 1e-3 \
      --collector gtl \
      --max_directions 200 \
      --fopng_lambda_reg 1e-3 \
      --fopng_epsilon 1e-4 \
      --batch_size 10 \
      --seed $seed \
      --device mps

  echo "Running SGD seed $seed"
  python3 main.py \
      --dataset rotated_mnist \
      --method sgd \
      --num_tasks 5 \
      --epochs 5 \
      --lr 1e-3 \
      --batch_size 10 \
      --seed $seed \
      --device mps
done
