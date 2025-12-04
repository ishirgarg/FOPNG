python3 main.py \
        --dataset rotated_mnist \
        --method adam \
        --num_tasks 5 \
        --grads_per_task 80 \
        --epochs 5 \
        --lr 5e-4 \
        --batch_size 10 \
        --seed 1 \
        --device cpu

python3 main.py \
      --dataset rotated_mnist \
      --method adam \
      --num_tasks 5 \
      --grads_per_task 80 \
      --epochs 5 \
      --lr 1e-4 \
      --batch_size 10 \
      --seed 1 \
      --device cpu