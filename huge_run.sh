lrs=(1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2)
for lr in ${lrs[@]}
do
    python3 main.py --dataset split_mnist_ic --method fopng --fisher diagonal --num_tasks 5 --epochs 5 --lr $lr --use_sgd --first_task_lr $lr --collector gtl --max_directions 2000 --grads_per_task 80 --fopng_lambda_reg 1e-3 --seed 1 --fisher_batch_size 1024
    python3 main.py --dataset split_mnist_ic --method fopng_prefisher --fisher diagonal --num_tasks 5 --epochs 5 --lr $lr --use_sgd --first_task_lr $lr --collector gtl --max_directions 2000 --grads_per_task 80 --fopng_lambda_reg 1e-3 --seed 1 --fisher_batch_size 1024

    python3 main.py --dataset rotated_mnist --method fopng --fisher diagonal --num_tasks 5 --epochs 5 --lr $lr --use_sgd --first_task_lr $lr --collector gtl --max_directions 2000 --grads_per_task 80 --fopng_lambda_reg 1e-3 --seed 1 --fisher_batch_size 1024
    python3 main.py --dataset rotated_mnist --method fopng_prefisher --fisher diagonal --num_tasks 5 --epochs 5 --lr $lr --use_sgd --first_task_lr $lr --collector gtl --max_directions 2000 --grads_per_task 80 --fopng_lambda_reg 1e-3 --seed 1 --fisher_batch_size 1024

    python3 main.py --dataset permuted_mnist --method fopng --fisher diagonal --num_tasks 5 --epochs 5 --lr $lr --use_sgd --first_task_lr $lr --collector gtl --max_directions 2000 --grads_per_task 80 --fopng_lambda_reg 1e-3 --seed 1 --fisher_batch_size 1024
    python3 main.py --dataset permuted_mnist --method fopng_prefisher --fisher diagonal --num_tasks 5 --epochs 5 --lr $lr --use_sgd --first_task_lr $lr --collector gtl --max_directions 2000 --grads_per_task 80 --fopng_lambda_reg 1e-3 --seed 1 --fisher_batch_size 1024
done

for lr in ${lrs[@]}
do
    python3 main.py --dataset split_cifar10 --method fopng --fisher diagonal --num_tasks 5 --epochs 5 --lr $lr --use_sgd --first_task_lr 1e-3 --collector gtl --max_directions 2000 --grads_per_task 80 --fopng_lambda_reg 1e-3 --seed 1 --fisher_batch_size 1024
    python3 main.py --dataset split_cifar10 --method fopng_prefisher --fisher diagonal --num_tasks 5 --epochs 5 --lr $lr --use_sgd --first_task_lr 1e-3 --collector gtl --max_directions 2000 --grads_per_task 80 --fopng_lambda_reg 1e-3 --seed 1 --fisher_batch_size 1024

    python3 main.py --dataset split_cifar100 --method fopng --fisher diagonal --num_tasks 10 --epochs 10 --lr $lr --use_sgd --first_task_lr 1e-2 --collector gtl --max_directions 2000 --grads_per_task 80 --fopng_lambda_reg 1e-3 --seed 1 --fisher_batch_size 1024
    python3 main.py --dataset split_cifar100 --method fopng_prefisher --fisher diagonal --num_tasks 5 --epochs 5 --lr $lr --use_sgd --first_task_lr 1e-2 --collector gtl --max_directions 2000 --grads_per_task 80 --fopng_lambda_reg 1e-3 --seed 1 --fisher_batch_size 1024
done