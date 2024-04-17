#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1

for i in $(seq 0 "$((${shards}-1))"); do
    echo "shard: $((${i}+1))/${shards}"
    r=0
    python sisa.py \
        --model mt5-base \
        --dataset data/xnli-en/datasetfile \
        --container "xnli-en-${shards}" \
        --shard "${i}" \
        --label "${r}" \
        --test \
        --batch_size 8 \
        --bf16_full_eval
done
