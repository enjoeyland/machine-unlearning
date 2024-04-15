#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1

for i in $(seq 0 "$((${shards}-1))"); do
    for j in {0..15}; do
        echo "shard: $((${i}+1))/${shards}, requests: $((${j}+1))/16"
        r=$((${j}*${shards}/5))
        python sisa.py --model mt5-base --train --slices 1 --dataset data/xnli-en/datasetfile --label "${r}" --epochs 3 --batch_size 8 --learning_rate 1e-5 --optimizer adam --chkpt_interval 1 --container "xnli-en-${shards}" --shard "${i}"
    done
done
