#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1
    
if [[ ! -d "containers/xnli-${shards}" ]] ; then
    mkdir "containers/xnli-en-${shards}"
    mkdir "containers/xnli-en-${shards}/cache"
    mkdir "containers/xnli-en-${shards}/times"
    mkdir "containers/xnli-en-${shards}/outputs"
    echo 0 > "containers/xnli-en-${shards}/times/null.time"
fi

python distribution.py --shards "${shards}" --distribution uniform --container "xnli-en-${shards}" --dataset data/xnli-en/datasetfile --label 0

for j in {1..15}; do
    r=$((${j}*${shards}/5))
    python distribution.py --requests "${r}" --distribution uniform --container "xnli-en-${shards}" --dataset data/xnli-en/datasetfile --label "${r}"
done
