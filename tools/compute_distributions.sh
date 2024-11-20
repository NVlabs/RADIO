#!/bin/bash
set -o xtrace

export PYTHONPATH=.:examples
function trun () {
    LOGLEVEL=WARNING NCCL_DEBUG=WARN torchrun --nproc_per_node=$SUBMIT_GPUS --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nnodes=$NUM_NODES --node_rank=$NODE_RANK $@
}

trun examples/feature_distribution.py --model-version openai_clip,ViT-L/14@336px --resolution 336 336 --output distributions/openai_clip.csv
trun examples/feature_distribution.py --model-version open_clip,ViT-H-14-378-quickgelu,dfn5b --resolution 378 378 --output distributions/dfn_clip.csv
trun examples/feature_distribution.py --model-version open_clip,ViT-SO400M-14-SigLIP-384,webli --resolution 384 384 --output distributions/siglip.csv
trun examples/feature_distribution.py --model-version dinov2_vitg14_reg --resolution 224 224 --output distributions/dinov2.csv
trun examples/feature_distribution.py --model-version sam,/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/model_zoo/sam/sam_vit_h_4b8939.pth --resolution 1024 1024 --output distributions/sam.csv --batch-size 16
