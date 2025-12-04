python3 main.py \
        --dataset permuted_mnist \
        --method fopng \
        --fisher diagonal \
        --num_tasks 5 \
        --epochs 5 \
        --lr 1e-4 \
        --collector gtl \
        --max_directions 2000 \
        --grads_per_task 150 \
        --fopng_lambda_reg 1e-3 \
        --fopng_new_fisher_weight 0.5 \
        --batch_size 10 \
        --seed 1 \
        --device mps

python3 main.py \
        --dataset permuted_mnist \
        --method fopng \
        --fisher diagonal \
        --num_tasks 5 \
        --epochs 5 \
        --lr 5e-4 \
        --collector gtl \
        --max_directions 2000 \
        --grads_per_task 150 \
        --fopng_lambda_reg 1e-3 \
        --fopng_new_fisher_weight 0.5 \
        --batch_size 10 \
        --seed 1 \
        --device mps

python3 main.py \
        --dataset permuted_mnist \
        --method fopng \
        --fisher diagonal \
        --num_tasks 5 \
        --epochs 5 \
        --lr 1e-3 \
        --collector gtl \
        --max_directions 2000 \
        --grads_per_task 150 \
        --fopng_lambda_reg 1e-3 \
        --fopng_new_fisher_weight 0.5 \
        --batch_size 10 \
        --seed 1 \
        --device mps

python3 main.py \
        --dataset permuted_mnist \
        --method fopng \
        --fisher diagonal \
        --num_tasks 5 \
        --epochs 5 \
        --lr 1e-4 \
        --collector gtl \
        --max_directions 2000 \
        --grads_per_task 150 \
        --fopng_lambda_reg 1e-3 \
        --fopng_new_fisher_weight 0.5 \
        --batch_size 10 \
        --seed 1 \
        --device mps

python3 main.py \
        --dataset permuted_mnist \
        --method fopng \
        --fisher diagonal \
        --num_tasks 5 \
        --epochs 5 \
        --lr 5e-4 \
        --collector gtl \
        --max_directions 2000 \
        --grads_per_task 150 \
        --fopng_lambda_reg 1e-3 \
        --fopng_new_fisher_weight 0.5 \
        --batch_size 10 \
        --seed 1 \
        --device mps

python3 main.py \
        --dataset rotated_mnist \
        --method fopng \
        --fisher diagonal \
        --num_tasks 5 \
        --epochs 5 \
        --lr 1e-3 \
        --collector gtl \
        --max_directions 2000 \
        --grads_per_task 150 \
        --fopng_lambda_reg 1e-2 \
        --fopng_new_fisher_weight 0.5 \
        --batch_size 10 \
        --seed 1 \
        --device mps

python3 main.py \
        --dataset rotated_mnist \
        --method fopng \
        --fisher diagonal \
        --num_tasks 5 \
        --epochs 5 \
        --lr 5e-3 \
        --collector gtl \
        --max_directions 2000 \
        --grads_per_task 150 \
        --fopng_lambda_reg 1e-2 \
        --fopng_new_fisher_weight 0.5 \
        --batch_size 10 \
        --seed 1 \
        --device mps

python3 main.py \
        --dataset rotated_mnist \
        --method fopng \
        --fisher diagonal \
        --num_tasks 5 \
        --epochs 5 \
        --lr 1e-3 \
        --collector gtl \
        --max_directions 2000 \
        --grads_per_task 150 \
        --fopng_lambda_reg 1e-4 \
        --fopng_new_fisher_weight 0.5 \
        --batch_size 10 \
        --seed 1 \
        --device mps

python3 main.py \
        --dataset rotated_mnist \
        --method fopng \
        --fisher diagonal \
        --num_tasks 5 \
        --epochs 5 \
        --lr 5e-3 \
        --collector gtl \
        --max_directions 2000 \
        --grads_per_task 150 \
        --fopng_lambda_reg 1e-4 \
        --fopng_new_fisher_weight 0.5 \
        --batch_size 10 \
        --seed 1 \
        --device mps