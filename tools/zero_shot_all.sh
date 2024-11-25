#!/bin/bash
# set -o xtrace

# Default values
CSV='zero_shot.csv'
PATCH_SIZE=16
MIN_RES=""
STEP=""
MAX_RES=""

# Parse command-line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --patch-size)
            PATCH_SIZE="$2"
            shift 2
            ;;
        --csv)
            CSV="$2"
            shift 2
            ;;
        --min-res)
            MIN_RES="$2"
            shift 2
            ;;
        --step)
            STEP="$2"
            shift 2
            ;;
        --max-res)
            MAX_RES="$2"
            shift 2
            ;;
        *)
            CHK="$1"
            shift
            ;;
    esac
done

if [ -z "$CHK" ]; then
    echo "Error: Model checkpoint must be specified."
    exit 1
fi

if [ -z "$MIN_RES" ]; then
    MIN_RES=$((256 * PATCH_SIZE / 16))
fi

if [ -z "$MAX_RES" ]; then
    MAX_RES=$((1024 * PATCH_SIZE / 16))
fi

if [ -z "$STEP" ]; then
    STEP=$((32 * PATCH_SIZE / 16))
fi

export PYTHONPATH=.:examples

# Calculate the sequence of resolutions based on the patch size
RESOLUTIONS=()
for res in $(seq $MIN_RES $STEP $MAX_RES); do
    RESOLUTIONS+=($res)
done

trun () {
    LOGLEVEL=WARNING NCCL_DEBUG=WARN torchrun --nproc_per_node=$SUBMIT_GPUS --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nnodes=$NUM_NODES --node_rank=$NODE_RANK "$@"
}

echo "Resolution,Top 1 Accuracy" >> "$CSV"

for res in "${RESOLUTIONS[@]}"; do
    echo "Resolution: $res $res"
    trun examples/zero_shot_imagenet.py --model-version "$CHK" --resolution "$res" "$res" --csv-out "$CSV" | grep "Top 1"
    echo "Resolution: $res"
    trun examples/zero_shot_imagenet.py --model-version "$CHK" --resolution "$res" --csv-out "$CSV" | grep "Top 1"
done

echo "Resolution: 1024 1024 - VitDet: 16"
trun examples/zero_shot_imagenet.py --model-version "$CHK" --resolution 1024 1024 --vitdet-window-size 16 --csv-out "$CSV" | grep "Top 1"

KNN_RESOLUTIONS=(512)

for res in "${KNN_RESOLUTIONS[@]}"; do
    echo "KNN Resolution: $res"
    trun examples/knn_classification.py --model-version "$CHK" --resolution 512 --dataset imagenet-1k | grep "Accuracy"
done
