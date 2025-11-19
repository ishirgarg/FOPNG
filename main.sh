CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --dataset split_mnist \
    --method sgd \
    --num_tasks 5 \
    --epochs 5 \
    --lr 1e-3 \
    --batch_size 10 \
    --seed 42 \
    --device cuda \
    > runs/run0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python3 main.py \
    --dataset split_mnist \
    --method ogd \
    --collector gtl \
    --max_directions 200 \
    --num_tasks 5 \
    --epochs 5 \
    --lr 1e-3 \
    --batch_size 10 \
    --seed 42 \
    --device cuda \
    > runs/run1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python3 main.py \
    --dataset split_mnist \
    --method fopng \
    --fisher diagonal \
    --collector gtl \
    --max_directions 200 \
    --num_tasks 5 \
    --epochs 5 \
    --lr 1e-3 \
    --batch_size 10 \
    --seed 42 \
    --fopng_lambda_reg 1e-3 \
    --fopng_epsilon 1e-4 \
    --device cuda \
    > runs/run2.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python3 main.py \
    --dataset split_mnist \
    --method fopng \
    --fisher diagonal \
    --collector ave \
    --max_directions 200 \
    --num_tasks 5 \
    --epochs 5 \
    --lr 1e-3 \
    --batch_size 10 \
    --seed 42 \
    --fopng_lambda_reg 1e-3 \
    --fopng_epsilon 1e-4 \
    --device cuda \
    > runs/run3.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 python3 main.py \
    --dataset split_mnist \
    --method fopng \
    --fisher diagonal \
    --collector gtl \
    --max_directions 200 \
    --num_tasks 5 \
    --epochs 5 \
    --lr 1e-3 \
    --batch_size 10 \
    --seed 42 \
    --fopng_lambda_reg 1e-4 \
    --fopng_epsilon 1e-4 \
    --device cuda \
    > runs/run4.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 python3 main.py \
    --dataset permuted_mnist \
    --method sgd \
    --num_tasks 5 \
    --epochs 5 \
    --lr 1e-3 \
    --batch_size 10 \
    --seed 42 \
    --device cuda \
    > runs/run5.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 python3 main.py \
    --dataset permuted_mnist \
    --method ogd \
    --collector gtl \
    --max_directions 200 \
    --num_tasks 5 \
    --epochs 5 \
    --lr 1e-3 \
    --batch_size 10 \
    --seed 42 \
    --device cuda \
    > runs/run6.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 python3 main.py \
    --dataset permuted_mnist \
    --method fopng \
    --fisher diagonal \
    --collector gtl \
    --max_directions 200 \
    --num_tasks 5 \
    --epochs 5 \
    --lr 1e-3 \
    --batch_size 10 \
    --seed 42 \
    --fopng_lambda_reg 1e-3 \
    --fopng_epsilon 1e-4 \
    --device cuda \
    > runs/run7.log 2>&1 &