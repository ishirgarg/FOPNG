#!/bin/bash
#SBATCH --job-name=fopng_sweep
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --array=0-15

############################################
# Parameters
############################################
lrs=(1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1)
datasets=(split_mnist_ic rotated_mnist permuted_mnist split_cifar10)
ewc_lambdas=(10 50 100 400)

NUM_WORKERS=16
WORKER_ID=${SLURM_ARRAY_TASK_ID}

############################################
# Build command list
############################################
COMMANDS=()

#!/bin/bash
#SBATCH --job-name=fopng_sweep
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --array=0-15

############################################
# Parameters
############################################
lrs=(1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1)
datasets=(split_mnist_ic rotated_mnist permuted_mnist split_cifar10)
ewc_lambdas=(10 50 100 400)

NUM_WORKERS=16
WORKER_ID=${SLURM_ARRAY_TASK_ID}

############################################
# Build command list
############################################
COMMANDS=()

# Small / CIFAR-10 style datasets
for dataset in "${datasets[@]}"; do

  # Dataset-specific first_task_lr
  if [[ "$dataset" == "split_cifar10" ]]; then
    FIRST_TASK_LR=1e-3
  else
    FIRST_TASK_LR=1e-2
  fi

  for lr in "${lrs[@]}"; do

    COMMANDS+=("python main.py --dataset $dataset --method fopng --fisher diagonal \
      --num_tasks 5 --epochs 5 --lr $lr --use_sgd \
      --first_task_lr $FIRST_TASK_LR --collector gtl \
      --max_directions 2000 --grads_per_task 80 \
      --fopng_lambda_reg 1e-3 --seed 1 \
      --fisher_batch_size 1024 --device cuda")

    COMMANDS+=("python main.py --dataset $dataset --method fopng_prefisher --fisher diagonal \
      --num_tasks 5 --epochs 5 --lr $lr --use_sgd \
      --first_task_lr $FIRST_TASK_LR --collector gtl \
      --max_directions 2000 --grads_per_task 80 \
      --fopng_lambda_reg 1e-3 --seed 1 \
      --fisher_batch_size 1024 --device cuda")

    COMMANDS+=("python main.py --dataset $dataset --method ogd \
      --num_tasks 5 --epochs 5 --lr $lr --use_sgd \
      --first_task_lr $FIRST_TASK_LR --batch_size 10 --max_directions 2000 \
      --grads_per_task 80 --seed 1 --device cuda")

    for lambda in "${ewc_lambdas[@]}"; do
      COMMANDS+=("python main.py --dataset $dataset --method ewc --fisher diagonal \
        --num_tasks 5 --epochs 5 --lr $lr --use_sgd \
        --first_task_lr $FIRST_TASK_LR --batch_size 10 \
        --ewc_lambda $lambda --seed 1 --device cuda")
    done

    COMMANDS+=("python main.py --dataset $dataset --method fng --fisher diagonal \
      --num_tasks 5 --epochs 5 --lr $lr --use_sgd \
      --first_task_lr $FIRST_TASK_LR --grads_per_task 80 \
      --fopng_lambda_reg 1e-3 --batch_size 10 \
      --seed 1 --device cuda")

    COMMANDS+=("python main.py --dataset $dataset --method adam \
      --num_tasks 5 --epochs 5 --lr $lr --use_sgd \
      --first_task_lr $FIRST_TASK_LR --batch_size 10 --seed 1 --device cuda")

    COMMANDS+=("python main.py --dataset $dataset --method sgd \
      --num_tasks 5 --epochs 5 --lr $lr --use_sgd \
      --first_task_lr $FIRST_TASK_LR --batch_size 10 --seed 1 --device cuda")
  done
done


# CIFAR-100
for lr in "${lrs[@]}"; do

  COMMANDS+=("python main.py --dataset split_cifar100 --method fopng --fisher diagonal \
    --num_tasks 10 --epochs 10 --lr $lr --use_sgd \
    --first_task_lr 1e-2 --collector gtl \
    --max_directions 2000 --grads_per_task 80 \
    --fopng_lambda_reg 1e-3 --seed 1 \
    --fisher_batch_size 1024 --device cuda")

  COMMANDS+=("python main.py --dataset split_cifar100 --method fopng_prefisher --fisher diagonal \
    --num_tasks 10 --epochs 10 --lr $lr --use_sgd \
    --first_task_lr 1e-2 --collector gtl \
    --max_directions 2000 --grads_per_task 80 \
    --fopng_lambda_reg 1e-3 --seed 1 \
    --fisher_batch_size 1024 --device cuda")

  COMMANDS+=("python main.py --dataset split_cifar100 --method ogd \
    --num_tasks 10 --epochs 10 --lr $lr --use_sgd \
    --first_task_lr 1e-2 --batch_size 10 --max_directions 2000 \
    --grads_per_task 80 --seed 1 --device cuda")

  for lambda in "${ewc_lambdas[@]}"; do
    COMMANDS+=("python main.py --dataset split_cifar100 --method ewc --fisher diagonal \
      --num_tasks 10 --epochs 10 --lr $lr --use_sgd \
      --first_task_lr 1e-2 --batch_size 10 \
      --ewc_lambda $lambda --seed 1 --device cuda")
  done

  COMMANDS+=("python main.py --dataset split_cifar100 --method fng --fisher diagonal \
    --num_tasks 10 --epochs 10 --lr $lr --use_sgd \
    --first_task_lr 1e-2 --grads_per_task 80 \
    --fopng_lambda_reg 1e-3 --batch_size 10 \
    --seed 1 --device cuda")

  COMMANDS+=("python main.py --dataset split_cifar100 --method adam \
    --num_tasks 10 --epochs 10 --lr $lr --use_sgd \
    --first_task_lr 1e-2 --batch_size 10 --seed 1 --device cuda")

  COMMANDS+=("python main.py --dataset split_cifar100 --method sgd \
    --num_tasks 10 --epochs 10 --lr $lr --use_sgd \
    --first_task_lr 1e-2 --batch_size 10 --seed 1 --device cuda")
done

############################################
# Sharded execution
############################################
TOTAL=${#COMMANDS[@]}

for ((i=WORKER_ID; i<TOTAL; i+=NUM_WORKERS)); do
  eval "${COMMANDS[$i]}"
done
