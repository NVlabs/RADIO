#!/bin/bash
# set -o xtrace

# Default values
CSV='zero_shot.csv'
PATCH_SIZE=16
MIN_RES=0
STEP=""
MAX_RES=100000

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

export PYTHONPATH=.:examples

# Calculate the sequence of resolutions based on the patch size
P16_RESOLUTIONS=(224 256 432 512 768 1024)
RESOLUTIONS=()
for res in ${P16_RESOLUTIONS[@]}; do
    new_res=$((res * PATCH_SIZE / 16))
    if [ $new_res -ge $MIN_RES ] && [ $new_res -le $MAX_RES ]; then
        RESOLUTIONS+=($new_res)
    fi
done

echo "Zero Shot Resolutions: ${RESOLUTIONS[@]}"

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

vdt_res=$((1024 * PATCH_SIZE / 16))
echo "Resolution: $vdt_res $vdt_res - VitDet: 16"
trun examples/zero_shot_imagenet.py --model-version "$CHK" --resolution $vdt_res $vdt_res --vitdet-window-size 16 --csv-out "$CSV" | grep "Top 1"

P16_KNN_RESOLUTIONS=(512)
KNN_RESOLUTIONS=()
for res in ${P16_KNN_RESOLUTIONS[@]}; do
    new_res=$((res * PATCH_SIZE / 16))
    KNN_RESOLUTIONS+=($new_res)
done

echo "KNN Resolutions: ${KNN_RESOLUTIONS[@]}"

for res in "${KNN_RESOLUTIONS[@]}"; do
    echo "KNN Resolution: $res"
    trun examples/knn_classification.py --model-version "$CHK" --resolution $res $res --dataset imagenet-1k | grep "Accuracy"
done
