#!/bin/bash
# set -o xtrace

GPUS=8
IMAGE="/lustre/fs6/portfolios/llmservice/users/mranzinger/sqsh_images/dler+evfm+radio+examples.sqsh"

submit() {
    submit_job --gpu $GPUS --partition batch_block1,batch_block3,batch_block4 --workdir `pwd` \
               --image $IMAGE --coolname --duration 1 --more_srun_args=--gpus-per-node=$GPUS \
               -c "export LOGLEVEL=WARNING; export NCCL_DEBUG=WARN; export PYTHONPATH=.:examples; torchrun --nproc_per_node=\$SUBMIT_GPUS --master_addr=\$MASTER_ADDR --master_port=\$MASTER_PORT --nnodes=\$NUM_NODES --node_rank=\$NODE_RANK $*"
}
# export PYTHONPATH=.:examples

RADIOv2="/lustre/fs6/portfolios/llmservice/users/mranzinger/output/evfm/ohem/2-8-24_vit-h-16_baseline/checkpoints/checkpoint-46.pth.tar"

submit examples/ssl_metrics.py --log-wandb --model-version radio_v1 --wandb-group RADIOv1 --resolution 224
submit examples/ssl_metrics.py --log-wandb --model-version radio_v1 --wandb-group RADIOv1 --resolution 336
submit examples/ssl_metrics.py --log-wandb --model-version radio_v1 --wandb-group RADIOv1 --resolution 378
submit examples/ssl_metrics.py --log-wandb --model-version radio_v1 --wandb-group RADIOv1 --resolution 518

submit examples/ssl_metrics.py --log-wandb --model-version $RADIOv2 --wandb-group RADIOv2 --wandb-job-type radio_v2_chk46_hires --resolution 224
submit examples/ssl_metrics.py --log-wandb --model-version $RADIOv2 --wandb-group RADIOv2 --wandb-job-type radio_v2_chk46_hires --resolution 336
submit examples/ssl_metrics.py --log-wandb --model-version $RADIOv2 --wandb-group RADIOv2 --wandb-job-type radio_v2_chk46_hires --resolution 384
submit examples/ssl_metrics.py --log-wandb --model-version $RADIOv2 --wandb-group RADIOv2 --wandb-job-type radio_v2_chk46_hires --resolution 432
submit examples/ssl_metrics.py --log-wandb --model-version $RADIOv2 --wandb-group RADIOv2 --wandb-job-type radio_v2_chk46_hires --resolution 512
submit examples/ssl_metrics.py --log-wandb --model-version $RADIOv2 --wandb-group RADIOv2 --wandb-job-type radio_v2_chk46_hires --resolution 1024 -b 1
submit examples/ssl_metrics.py --log-wandb --model-version $RADIOv2 --wandb-group RADIOv2 --wandb-job-type radio_v2_chk46_hires --resolution 1024 -b 1 --vitdet-window-size 8
submit examples/ssl_metrics.py --log-wandb --model-version $RADIOv2 --wandb-group RADIOv2 --wandb-job-type radio_v2_chk46_hires --resolution 1024 -b 1 --vitdet-window-size 16

submit examples/ssl_metrics.py --log-wandb --model-version dinov2_vits14_reg --resolution 224
submit examples/ssl_metrics.py --log-wandb --model-version dinov2_vitb14_reg --resolution 224
submit examples/ssl_metrics.py --log-wandb --model-version dinov2_vitl14_reg --resolution 224
submit examples/ssl_metrics.py --log-wandb --model-version dinov2_vitg14_reg --resolution 224 -b 1

submit examples/ssl_metrics.py --log-wandb --model-version "open_clip,ViT-H-14-quickgelu,dfn5b" --resolution 224
submit examples/ssl_metrics.py --log-wandb --model-version "open_clip,ViT-H-14-378-quickgelu,dfn5b" --resolution 378
submit examples/ssl_metrics.py --log-wandb --model-version "open_clip,ViT-H-14-quickgelu,metaclip_fullcc" --resolution 224
