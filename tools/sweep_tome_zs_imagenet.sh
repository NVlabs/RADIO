#!/usr/bin/env bash
set -euo pipefail

# --- Config: adjust if needed ---
MODEL="/lustre/fsw/portfolios/llmservice/users/mranzinger/output/evfm/commercial/v3/tome/vit-l-16_s4-rtx_p0.5-l0.1-h0.9_side/checkpoints/last.pth.tar"
CSV_OUT="tome_zero_shot.csv"
RES_W=256
RES_H=256
MODE="CONSTANT"
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
  echo "ToMe Mode,Type,Reduction,Resolution,Accuracy,Elapsed" > "$CSV_OUT"
fi

# Sweep r_pct from 0.00 to 0.90 inclusive in 0.01 increments
for i in {0..90}; do
  r_pct=$(printf "0.%02d" "$i")
  echo "Running r_pct=${r_pct} ..."
  trun examples/zero_shot_imagenet.py \
    --model-version "$MODEL" \
    --resolution "$RES_W" "$RES_H" \
    --use-local-lib \
    --tome-config "mode=${MODE},r_pct=${r_pct}" \
    --csv-out "$CSV_OUT" \
    --csv-exp-name "constant,r_pct,${r_pct}"
done

echo "Sweep complete. Results appended to ${CSV_OUT}."
