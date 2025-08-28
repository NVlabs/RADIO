#!/usr/bin/env bash
set -euo pipefail

# --- Config: adjust if needed ---
MODEL="/lustre/fsw/portfolios/llmservice/users/mranzinger/output/evfm/commercial/v3/tome/vit-l-16_s4-rtx_p0.5-l0.1-h0.9_side/checkpoints/last.pth.tar"
CSV_OUT="tome_zero_shot_constant.csv"
RES_W=256
RES_H=256
MODE="CONSTANT"
R_TYPE="r_pct"
START=0.0
END=0.9
INCREMENT=0.01
BATCH_SIZE=128
# --------------------------------

# Parse command-line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --csv)
            CSV_OUT="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --adaptor)
            ADAPTOR="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --start)
            START="$2"
            shift 2
            ;;
        --end)
            END="$2"
            shift 2
            ;;
        --increment)
            INCREMENT="$2"
            shift 2
            ;;
        --r-type)
            R_TYPE="$2"
            shift 2
            ;;
        *)
            MODEL="$1"
            shift
            ;;
    esac
done

export PYTHONPATH=.:examples

trun () {
    LOGLEVEL=WARNING NCCL_DEBUG=WARN torchrun --nproc_per_node=$SUBMIT_GPUS --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nnodes=$NUM_NODES --node_rank=$NODE_RANK "$@"
}

# Create CSV with header if it doesn't exist
if [[ ! -f "$CSV_OUT" ]]; then
  echo "ToMe Mode,Type,Reduction,Resolution,Accuracy,Elapsed,Avg Tokens" > "$CSV_OUT"
fi

mode_lower=$(echo "$MODE" | tr '[:upper:]' '[:lower:]')

# Sweep using floating-point range (e.g., 0.0 to 1.5 in 0.1 increments)
for r_pct in $(seq "$START" "$INCREMENT" "$END"); do
    echo "Running ${R_TYPE}=${r_pct} ..."
    trun examples/zero_shot_imagenet.py \
        --model-version "$MODEL" \
        --resolution "$RES_W" "$RES_H" \
        --batch-size $BATCH_SIZE \
        --use-local-lib \
        --tome-config "mode=${MODE},${R_TYPE}=${r_pct}" \
        --csv-out "$CSV_OUT" \
        --csv-exp-name "${mode_lower},${R_TYPE},${r_pct}"
done

echo "Sweep complete. Results appended to ${CSV_OUT}."
