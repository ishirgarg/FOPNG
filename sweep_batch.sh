#!/bin/bash
seeds=$(seq 2 5)
bs=(1 2 4 8 16 32 64 128 256 512 768 1024 2048)

# ---------------------- FOPNG ----------------------
for seed in $seeds; do
    for b in "${bs[@]}"; do
        echo "Running FOPNG rotated_mnist (fisher_batch_size=$b, seed=$seed)"
        python3 main.py \
            --dataset rotated_mnist \
            --method fopng \
            --fisher diagonal \
            --num_tasks 5 \
            --epochs 5 \
            --lr 1e-3 \
            --collector gtl \
            --max_directions 2000 \
            --grads_per_task 80 \
            --fopng_lambda_reg 1e-2 \
            --fopng_new_fisher_weight 0.5 \
            --fisher_batch_size "$b" \
            --batch_size 10 \
            --seed "$seed" \
            --device mps
    done
done

for seed in $seeds; do
    for b in "${bs[@]}"; do
        echo "Running FOPNG permuted_mnist (fisher_batch_size=$b, seed=$seed)"
        python3 main.py \
            --dataset permuted_mnist \
            --method fopng \
            --fisher diagonal \
            --num_tasks 5 \
            --epochs 5 \
            --lr 1e-4 \
            --collector gtl \
            --max_directions 2000 \
            --grads_per_task 80 \
            --fopng_lambda_reg 1e-3 \
            --fopng_new_fisher_weight 0.5 \
            --fisher_batch_size "$b" \
            --batch_size 10 \
            --seed "$seed" \
            --device mps
    done
done