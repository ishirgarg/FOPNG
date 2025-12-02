#!/bin/bash
# Quick test script for EWC

echo "Testing EWC on 2 tasks..."

python3 main.py \
    --dataset permuted_mnist \
    --method ewc \
    --fisher diagonal \
    --num_tasks 2 \
    --epochs 2 \
    --lr 1e-3 \
    --ewc_lambda 5000 \
    --batch_size 10 \
    --seed 42 \
    --device auto \
    --no_wandb

echo ""
echo "Test complete! If no errors, run: ./ewc.sh"

