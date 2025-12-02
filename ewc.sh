#!/bin/bash

datasets=(rotated_mnist split_mnist permuted_mnist)
seeds=$(seq 1 1)

# EWC hyperparameter sweeps (original paper range)
# Lambda values - paper uses 1-1000 range typically
ewc_lambdas=(10 50 100 400)
# Learning rates
lrs_ewc=(5e-4 1e-3 5e-3 1e-2)

for dataset in "${datasets[@]}"; do
  for seed in $seeds; do
    echo "==== DATASET $dataset | SEED $seed ===="

    # ---------------------- EWC ----------------------
    for ewc_lambda in "${ewc_lambdas[@]}"; do
      for lr in "${lrs_ewc[@]}"; do
        echo "Running EWC (lambda=$ewc_lambda, lr=$lr, seed=$seed) on $dataset"
        python3 main.py \
            --dataset "$dataset" \
            --method ewc \
            --fisher diagonal \
            --num_tasks 5 \
            --epochs 5 \
            --lr "$lr" \
            --ewc_lambda "$ewc_lambda" \
            --batch_size 10 \
            --seed "$seed" \
            --device auto \
            --wandb_project "fopng" \
            --wandb_tags ewc sweep "$dataset"
      done
    done
  done
done

