#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

container=$1
shards=5

for i in $(seq 0 2); do
    echo "shard: $((${i}+1))/${shards}"
    r=0
    python sisa.py \
        --model mt5-base \
        --dataset data/xnli-en/datasetfile \
        --container "xnli-en-${container}" \
        --shard "${i}" \
        --label "${r}" \
        --max_length 512 \
        --seed 42 \
        --train \
        --bf16 \
        --optimizer adamw \
        --learning_rate 5e-5 \
        --batch_size 8 \
        --gradient_accumulation_steps 4 \
        --logging_steps 100 \
        --evaluation_strategy epoch \
        --bf16_full_eval \
        --output_type argmax \
        --load_best_model_at_end \
        --epochs 3 \
        --slices 5 
done