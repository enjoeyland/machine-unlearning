#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

container=$1

python original.py \
    --model mt5-base \
    --dataset data/xnli-en/datasetfile \
    --name origianl \
    --container "xnli-en-${container}" \
    --max_length 512 \
    --seed 42 \
    --train \
    --bf16 \
    --optimizer adamw \
    --learning_rate 5e-5 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --logging_steps 100 \
    --eval_steps 500 \
    --bf16_full_eval \
    --output_type argmax \
    --load_best_model_at_end \
    --epochs 3
done
