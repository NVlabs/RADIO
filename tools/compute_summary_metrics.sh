#!/bin/bash
set -o xtrace

export PYTHONPATH=.:examples
function trun () {
    LOGLEVEL=WARNING NCCL_DEBUG=WARN torchrun --nproc_per_node=$SUBMIT_GPUS --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nnodes=$NUM_NODES --node_rank=$NODE_RANK $@
}

resolutions=(256 432 512 768 1024)
# resolutions=(256 432 512)
# resolutions=(768 1024)

echo "Zero Shot"
for res in "${resolutions[@]}"
do
    echo "Resolution $res"
    trun examples/zero_shot_imagenet.py --model-version $1 --resolution $res $res --batch-size 128 --dataset imagenet-1k
done

resolutions=(256 432 512)

echo "kNN Classification"
for res in "${resolutions[@]}"
do
    echo "Resolution $res"
    trun examples/knn_classification.py --model-version $1 --resolution $res $res --batch-size 64 --dataset imagenet-1k
done
