#!/bin/bash
# set -o xtrace

CHK="$1"
CSV="${2:-'zero_shot.csv'}"

export PYTHONPATH=.:examples

RESOLUTIONS=(224 256 336 432 512 768 1024)
# RESOLUTIONS=()
# for res in $(seq 224 32 1024); do
#     RESOLUTIONS+=($res)
# done

trun () {
    LOGLEVEL=WARNING NCCL_DEBUG=WARN torchrun --nproc_per_node=$SUBMIT_GPUS --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nnodes=$NUM_NODES --node_rank=$NODE_RANK $@ #2>&1
}

echo "Resolution,Top 1 Accuracy" > $CSV

for res in ${RESOLUTIONS[@]}; do
    echo "Resolution: $res $res"
    trun examples/zero_shot_imagenet.py --model-version $@ --resolution $res $res --csv-out $CSV | grep "Top 1"
    echo "Resolution: $res"
    trun examples/zero_shot_imagenet.py --model-version $@ --resolution $res --csv-out $CSV | grep "Top 1"
done

echo "Resolution: 1024 1024 - VitDet: 16"
trun examples/zero_shot_imagenet.py --model-version $@ --resolution 1024 1024 --vitdet-window-size 16 | grep "Top 1"

KNN_RESOLUTIONS=(512)

for res in ${KNN_RESOLUTIONS[@]}; do
    echo "KNN Resolution: $res"
    trun examples/knn_classification.py --model-version $@ --resolution 512 --dataset imagenet-1k | grep "Accuracy"
done
