#!/bin/bash
set -o xtrace

ROOT="vis_denoise"
N="128"

DATASET="/lustre/fs6/portfolios/llmservice/users/mranzinger/data/radio/paper_images"
# RADIOV2="/lustre/fs6/portfolios/llmservice/users/mranzinger/output/evfm/ohem/3-13-24_vit-h-16_bf16_ep50/checkpoints/radio_v2.1_bf16.pth.tar"
# RADIOV2="/lustre/fs6/portfolios/llmservice/users/mranzinger/output/evfm/ohem/3-9-24_vit-h-16_bf16_ep10/checkpoints/radio_v2.1_bf16.pth.tar"
RADIOV2="radio_v2.1"
# COMM_RADIO="/lustre/fs6/portfolios/llmservice/users/mranzinger/output/evfm/commercial_iad/n16_4-14-24_vit-h-16_dfn-siglip_low-res_ep150/checkpoints/checkpoint-113.pth.tar"
COMM_RADIO="/lustre/fs6/portfolios/llmservice/users/mranzinger/output/evfm/commercial/ord/dual-res/checkpoint-77.pth.tar"

E_RADIO="/lustre/fs6/portfolios/llmservice/users/mranzinger/output/evfm/eradio/n8_3-25-24_eradio_stage3-alt_s2ep77/checkpoints/eradio_v2.pth.tar"

export PYTHONPATH=.:examples

# python examples/visualize_features.py --dataset $DATASET --model-version openai_clip,ViT-L/14@336px -n $N --resolution 336 336 --output-dir $ROOT/openai_clip/vit-l/336

# python examples/visualize_features.py --dataset $DATASET --model-version open_clip,ViT-H-14-378-quickgelu,dfn5b -n $N --resolution 378 378 --output-dir $ROOT/open_clip/dfn5b/378

# python examples/visualize_features.py --dataset $DATASET --model-version dinov2_vitg14_reg -n $N --resolution 224 --patch-size 14 --resize-multiple 14 --output-dir $ROOT/dinov2_g_reg/224min --batch-size 1
# python examples/visualize_features.py --dataset $DATASET --model-version dinov2_vitg14_reg -n $N --resolution 378 --patch-size 14 --resize-multiple 14 --output-dir $ROOT/dinov2_g_reg/378min --batch-size 1
# python examples/visualize_features.py --dataset $DATASET --model-version dinov2_vitg14_reg -n $N --resolution 518 --patch-size 14 --resize-multiple 14 --output-dir $ROOT/dinov2_g_reg/518min --batch-size 1
# python examples/visualize_features.py --dataset $DATASET --model-version dinov2_vitg14_reg -n $N --resolution 896 --patch-size 14 --resize-multiple 14 --output-dir $ROOT/dinov2_g_reg/896min --batch-size 1
# python examples/visualize_features.py --dataset $DATASET --model-version dinov2_vitg14_reg -n $N --resolution 1022 --patch-size 14 --resize-multiple 14 --output-dir $ROOT/dinov2_g_reg/1022min --batch-size 1
# python examples/visualize_features.py --dataset $DATASET --model-version dinov2_vitg14_reg -n $N --resolution 1022 --patch-size 14 --resize-multiple 14 --max-dim --output-dir $ROOT/dinov2_g_reg/1022max --batch-size 1
# python examples/visualize_features.py --dataset $DATASET --model-version dinov2_vitg14_reg -n $N --resolution 2044 --patch-size 14 --resize-multiple 14 --max-dim --output-dir $ROOT/dinov2_g_reg/2044max --batch-size 1

# python examples/visualize_features.py --dataset $DATASET --model-version dinov2_vitg14_reg -n $N --resolution 224 224 --patch-size 14 --resize-multiple 14 --output-dir $ROOT/dinov2_g_reg/224 --batch-size 1
# python examples/visualize_features.py --dataset $DATASET --model-version dinov2_vitg14_reg -n $N --resolution 336 336 --patch-size 14 --resize-multiple 14 --output-dir $ROOT/dinov2_g_reg/336 --batch-size 1
# python examples/visualize_features.py --dataset $DATASET --model-version dinov2_vitg14_reg -n $N --resolution 518 518 --patch-size 14 --resize-multiple 14 --output-dir $ROOT/dinov2_g_reg/518 --batch-size 1
# python examples/visualize_features.py --dataset $DATASET --model-version dinov2_vitg14_reg -n $N --resolution 1022 1022 --patch-size 14 --resize-multiple 14 --output-dir $ROOT/dinov2_g_reg/1022 --batch-size 1

# python examples/visualize_features.py --dataset $DATASET --model-version radio_v1 -n $N --resolution 378 --patch-size 14 --resize-multiple 14 --output-dir $ROOT/radiov1/378


# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 224 --patch-size 16 --resize-multiple 16 --output-dir $ROOT/radiov2/224min

# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 256 --patch-size 16 --resize-multiple 16 --output-dir $ROOT/radiov2/256min

# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 336 336 --patch-size 16 --resize-multiple 16 --output-dir $ROOT/radiov2/336
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 336 --patch-size 16 --resize-multiple 16 --output-dir $ROOT/radiov2/336min
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 336 --patch-size 16 --resize-multiple 16 --max-dim --output-dir $ROOT/radiov2/336max
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 336 336 --patch-size 16 --resize-multiple 16 --max-dim --output-dir $ROOT/radiov2/336max-pad

# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 256 256   --patch-size 16 --resize-multiple 16 --adaptor-name dino_v2 --output-dir $ROOT/radiov2/256_dinov2
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 336 336   --patch-size 16 --resize-multiple 16 --adaptor-name dino_v2 --output-dir $ROOT/radiov2/336_dinov2
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 432 432   --patch-size 16 --resize-multiple 16 --adaptor-name dino_v2 --output-dir $ROOT/radiov2/432_dinov2
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 512 512   --patch-size 16 --resize-multiple 16 --adaptor-name dino_v2 --output-dir $ROOT/radiov2/512_dinov2
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 688 688   --patch-size 16 --resize-multiple 16 --adaptor-name dino_v2 --output-dir $ROOT/radiov2/688_dinov2
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 704 704   --patch-size 16 --resize-multiple 16 --adaptor-name dino_v2 --output-dir $ROOT/radiov2/704_dinov2
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 720 720   --patch-size 16 --resize-multiple 16 --adaptor-name dino_v2 --output-dir $ROOT/radiov2/720_dinov2
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 736 736   --patch-size 16 --resize-multiple 16 --adaptor-name dino_v2 --output-dir $ROOT/radiov2/736_dinov2
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 1024 1024 --patch-size 16 --resize-multiple 16 --adaptor-name dino_v2 --output-dir $ROOT/radiov2/1024_dinov2 --batch-size 1

# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 256 256   --patch-size 16 --resize-multiple 16 --adaptor-name sam --output-dir $ROOT/radiov2/256_sam
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 512 512   --patch-size 16 --resize-multiple 16 --adaptor-name sam --output-dir $ROOT/radiov2/512_sam
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 1024 1024 --patch-size 16 --resize-multiple 16 --adaptor-name sam --output-dir $ROOT/radiov2/1024_sam
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 2048 2048 --patch-size 16 --resize-multiple 16 --adaptor-name sam --output-dir $ROOT/radiov2/2048_sam

# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 432 432 --patch-size 16 --resize-multiple 16 --output-dir $ROOT/radiov2/432
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 432 --patch-size 16 --resize-multiple 16 --output-dir $ROOT/radiov2/432min
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 432 --patch-size 16 --resize-multiple 16 --max-dim --output-dir $ROOT/radiov2/432max
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 432 432 --patch-size 16 --resize-multiple 16 --max-dim --output-dir $ROOT/radiov2/432max-pad

# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 432 --patch-size 16 --resize-multiple 16 --output-dir $ROOT/radiov2/432min_clip --adaptor-name clip
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 432 --patch-size 16 --resize-multiple 16 --output-dir $ROOT/radiov2/432min_dinov2 --adaptor-name dino_v2
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 432 --patch-size 16 --resize-multiple 16 --max-dim --output-dir $ROOT/radiov2/432max_rtx --adaptor-name rtx-translate
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 432 432 --patch-size 16 --resize-multiple 16 --max-dim --output-dir $ROOT/radiov2/432max-pad_rtx --adaptor-name rtx-translate


# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 512 512 --patch-size 16 --resize-multiple 16 --output-dir $ROOT/radiov2/512
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 512 --patch-size 16 --resize-multiple 16 --output-dir $ROOT/radiov2/512min
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 512 --patch-size 16 --resize-multiple 16 --max-dim --output-dir $ROOT/radiov2/512max
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 512 512 --patch-size 16 --resize-multiple 16 --max-dim --output-dir $ROOT/radiov2/512max-pad

# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 512 --patch-size 16 --resize-multiple 16 --output-dir $ROOT/radiov2/512min_dinov2 --adaptor-name dino_v2
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 512 --patch-size 16 --resize-multiple 16 --output-dir $ROOT/radiov2/512min_clip --adaptor-name clip
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 512 --patch-size 16 --resize-multiple 16 --output-dir $ROOT/radiov2/512min_sam --adaptor-name sam


# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 768 --patch-size 16 --resize-multiple 16 --output-dir $ROOT/radiov2/768min

# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 1024 --patch-size 16 --resize-multiple 16 --output-dir $ROOT/radiov2/1024min --batch-size 1
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 1024 --patch-size 16 --resize-multiple 16 --output-dir $ROOT/radiov2/1024min_sam --batch-size 1 --adaptor-name sam
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 1024 --patch-size 16 --resize-multiple 16 --max-dim --output-dir $ROOT/radiov2/1024max --batch-size 1
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 1024 1024 --patch-size 16 --resize-multiple 16 --max-dim --output-dir $ROOT/radiov2/1024max-pad --batch-size 1
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 1024 1024 --patch-size 16 --resize-multiple 16 --output-dir $ROOT/radiov2/1024 --batch-size 1
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 2048 2048 --patch-size 16 --resize-multiple 16 --output-dir $ROOT/radiov2/2048 --batch-size 1

# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 1024 --vitdet-window-size 16 --patch-size 16 --resize-multiple 256 --output-dir $ROOT/radiov2/1024min_wnd-16
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 1024 --vitdet-window-size 16 --patch-size 16 --resize-multiple 256 --max-dim --output-dir $ROOT/radiov2/1024max_wnd-16
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 1024 1024 --vitdet-window-size 16 --patch-size 16 --resize-multiple 256 --max-dim --output-dir $ROOT/radiov2/1024max-pad_wnd-16

# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 1024 --vitdet-window-size 16 --patch-size 16 --resize-multiple 256 --output-dir $ROOT/radiov2/1024min_wnd-16_sam --adaptor-name sam
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 1024 1024 --vitdet-window-size 16 --patch-size 16 --resize-multiple 256 --max-dim --output-dir $ROOT/radiov2/1024max-pad_wnd-16_sam --adaptor-name sam
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 1024 --vitdet-window-size 16 --patch-size 16 --resize-multiple 256 --output-dir $ROOT/radiov2/1024min_wnd-16_dinov2 --adaptor-name dino_v2
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 1024 --vitdet-window-size 16 --patch-size 16 --resize-multiple 256 --output-dir $ROOT/radiov2/1024min_wnd-16_clip --adaptor-name clip

# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 1280 --patch-size 16 --resize-multiple 256 --output-dir $ROOT/radiov2/1280min
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 1280 --vitdet-window-size 16 --patch-size 16 --resize-multiple 256 --output-dir $ROOT/radiov2/1280min_wnd-16
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 1280 --vitdet-window-size 16 --patch-size 16 --resize-multiple 256 --output-dir $ROOT/radiov2/1280min_wnd-16_sam --adaptor-name sam

# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 2048 --patch-size 16 --resize-multiple 16 --max-dim --output-dir $ROOT/radiov2/2048max
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 2048 --vitdet-window-size 8 --patch-size 16 --resize-multiple 128 --max-dim --output-dir $ROOT/radiov2/2048max_wnd-8
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 2048 --vitdet-window-size 16 --patch-size 16 --resize-multiple 256 --max-dim --output-dir $ROOT/radiov2/2048max_wnd-16
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 2048 --vitdet-window-size 32 --patch-size 16 --resize-multiple 512 --max-dim --output-dir $ROOT/radiov2/2048max_wnd-32

# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 2048 --patch-size 16 --max-dim --output-dir $ROOT/radiov2/2048max_sam --adaptor-name sam
# python examples/visualize_features.py --dataset $DATASET --model-version $RADIOV2 -n $N --resolution 2048 --vitdet-window-size 16 --patch-size 16 --resize-multiple 256 --max-dim --output-dir $ROOT/radiov2/2048max_wnd-16_sam --adaptor-name sam


# python examples/visualize_features.py --dataset $DATASET --model-version $COMM_RADIO -n $N --resolution 256 --patch-size 16 --resize-multiple 16 --output-dir $ROOT/comm_radio/256min
# python examples/visualize_features.py --dataset $DATASET --model-version $COMM_RADIO -n $N --resolution 432 --patch-size 16 --resize-multiple 16 --output-dir $ROOT/comm_radio/432min
# python examples/visualize_features.py --dataset $DATASET --model-version $COMM_RADIO -n $N --resolution 512 --patch-size 16 --resize-multiple 16 --output-dir $ROOT/comm_radio/512min
# python examples/visualize_features.py --dataset $DATASET --model-version $COMM_RADIO -n $N --resolution 768 --patch-size 16 --resize-multiple 16 --output-dir $ROOT/comm_radio/768min
# python examples/visualize_features.py --dataset $DATASET --model-version $COMM_RADIO -n $N --resolution 1024 --patch-size 16 --resize-multiple 16 --output-dir $ROOT/comm_radio/1024min
# python examples/visualize_features.py --dataset $DATASET --model-version $COMM_RADIO -n $N --resolution 2048 --patch-size 16 --resize-multiple 16 --max-dim --output-dir $ROOT/comm_radio/2048max
# python examples/visualize_features.py --dataset $DATASET --model-version $COMM_RADIO -n $N --resolution 2048 --patch-size 16 --resize-multiple 16 --max-dim --output-dir $ROOT/comm_radio/2048max_clip --adaptor-name clip
# python examples/visualize_features.py --dataset $DATASET --model-version $COMM_RADIO -n $N --resolution 2048 --patch-size 16 --resize-multiple 16 --max-dim --output-dir $ROOT/comm_radio/2048max_dino --adaptor-name dino_v2
# python examples/visualize_features.py --dataset $DATASET --model-version $COMM_RADIO -n $N --resolution 2048 --patch-size 16 --resize-multiple 16 --max-dim --output-dir $ROOT/comm_radio/2048max_sam --adaptor-name sam


# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-b -n $N --resolution 256 256   --adaptor-name dino_v2 --output-dir $ROOT/radiov2.5/vit-b/256_dinov2
# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-b -n $N --resolution 432 432   --adaptor-name dino_v2 --output-dir $ROOT/radiov2.5/vit-b/432_dinov2
# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-b -n $N --resolution 512 512   --adaptor-name dino_v2 --output-dir $ROOT/radiov2.5/vit-b/512_dinov2
# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-b -n $N --resolution 688 688   --adaptor-name dino_v2 --output-dir $ROOT/radiov2.5/vit-b/688_dinov2
# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-b -n $N --resolution 704 704   --adaptor-name dino_v2 --output-dir $ROOT/radiov2.5/vit-b/704_dinov2
# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-b -n $N --resolution 720 720   --adaptor-name dino_v2 --output-dir $ROOT/radiov2.5/vit-b/720_dinov2
# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-b -n $N --resolution 736 736   --adaptor-name dino_v2 --output-dir $ROOT/radiov2.5/vit-b/736_dinov2
# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-b -n $N --resolution 1024 1024 --adaptor-name dino_v2 --output-dir $ROOT/radiov2.5/vit-b/1024_dinov2


# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-l -n $N --resolution 256 256   --adaptor-name dino_v2 --output-dir $ROOT/radiov2.5/vit-l/256_dinov2
# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-l -n $N --resolution 432 432   --adaptor-name dino_v2 --output-dir $ROOT/radiov2.5/vit-l/432_dinov2
# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-l -n $N --resolution 512 512   --adaptor-name dino_v2 --output-dir $ROOT/radiov2.5/vit-l/512_dinov2
# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-l -n $N --resolution 688 688   --adaptor-name dino_v2 --output-dir $ROOT/radiov2.5/vit-l/688_dinov2
# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-l -n $N --resolution 704 704   --adaptor-name dino_v2 --output-dir $ROOT/radiov2.5/vit-l/704_dinov2
# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-l -n $N --resolution 720 720   --adaptor-name dino_v2 --output-dir $ROOT/radiov2.5/vit-l/720_dinov2
# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-l -n $N --resolution 736 736   --adaptor-name dino_v2 --output-dir $ROOT/radiov2.5/vit-l/736_dinov2
# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-l -n $N --resolution 1024 1024 --adaptor-name dino_v2 --output-dir $ROOT/radiov2.5/vit-l/1024_dinov2

# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-l -n $N --resolution 256 256   --adaptor-name sam --output-dir $ROOT/radiov2.5/vit-l/256_sam
# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-l -n $N --resolution 512 512   --adaptor-name sam --output-dir $ROOT/radiov2.5/vit-l/512_sam
# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-l -n $N --resolution 1024 1024 --adaptor-name sam --output-dir $ROOT/radiov2.5/vit-l/1024_sam
# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-l -n $N --resolution 2048 2048 --adaptor-name sam --output-dir $ROOT/radiov2.5/vit-l/2048_sam

python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-h -n $N --resolution 256 --output-dir $ROOT/radiov2.5/vit-h/256min
python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-h -n $N --resolution 512 --output-dir $ROOT/radiov2.5/vit-h/512min
python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-h -n $N --resolution 768 --output-dir $ROOT/radiov2.5/vit-h/768min
python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-h -n $N --resolution 1024 --output-dir $ROOT/radiov2.5/vit-h/1024min
python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-h -n $N --resolution 2048 --output-dir $ROOT/radiov2.5/vit-h/2048min

# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-h -n $N --resolution 256 256   --adaptor-name dino_v2 --output-dir $ROOT/radiov2.5/vit-h/256_dinov2
# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-h -n $N --resolution 432 432   --adaptor-name dino_v2 --output-dir $ROOT/radiov2.5/vit-h/432_dinov2
# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-h -n $N --resolution 512 512   --adaptor-name dino_v2 --output-dir $ROOT/radiov2.5/vit-h/512_dinov2
# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-h -n $N --resolution 688 688   --adaptor-name dino_v2 --output-dir $ROOT/radiov2.5/vit-h/688_dinov2
# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-h -n $N --resolution 704 704   --adaptor-name dino_v2 --output-dir $ROOT/radiov2.5/vit-h/704_dinov2
# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-h -n $N --resolution 720 720   --adaptor-name dino_v2 --output-dir $ROOT/radiov2.5/vit-h/720_dinov2
# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-h -n $N --resolution 736 736   --adaptor-name dino_v2 --output-dir $ROOT/radiov2.5/vit-h/736_dinov2
# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-h -n $N --resolution 1024 1024 --adaptor-name dino_v2 --output-dir $ROOT/radiov2.5/vit-h/1024_dinov2

# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-h -n $N --resolution 256 256   --adaptor-name sam --output-dir $ROOT/radiov2.5/vit-h/256_sam
# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-h -n $N --resolution 512 512   --adaptor-name sam --output-dir $ROOT/radiov2.5/vit-h/512_sam
# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-h -n $N --resolution 1024 1024 --adaptor-name sam --output-dir $ROOT/radiov2.5/vit-h/1024_sam
# python examples/visualize_features.py --dataset $DATASET --model-version radio_v2.5-h -n $N --resolution 2048 2048 --adaptor-name sam --output-dir $ROOT/radiov2.5/vit-h/2048_sam


# python examples/visualize_features.py --dataset $DATASET --model-version sam,/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/model_zoo/sam/sam_vit_h_4b8939.pth -n $N --resolution 1024 1024 --output-dir $ROOT/sam_h
# python examples/visualize_features.py --dataset $DATASET --model-version sam,/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/model_zoo/sam/sam_vit_h_4b8939.pth -n $N --resolution 1024 1024 --max-dim --output-dir $ROOT/sam_h_max-pad


# python examples/visualize_features.py --dataset $DATASET --model-version open_clip,ViT-L-14,openai -n $N --resolution 224 224 --patch-size 14 --output-dir $ROOT/open_clip/openai/224
# python examples/visualize_features.py --dataset $DATASET --model-version open_clip,ViT-L-14-336,openai -n $N --resolution 336 336 --patch-size 14 --output-dir $ROOT/open_clip/openai/336


# python examples/visualize_features.py --dataset $DATASET --model-version open_clip,ViT-H-14,laion2b_s32b_b79k -n $N --resolution 224 224 --patch-size 14 --output-dir $ROOT/open_clip/openclip_vith/224


# python examples/visualize_features.py --dataset $DATASET --model-version open_clip,ViT-H-14-quickgelu,metaclip_fullcc -n $N --resolution 224 224 --patch-size 14 --output-dir $ROOT/open_clip/metaclip/224


# python examples/visualize_features.py --dataset $DATASET --model-version open_clip,ViT-SO400M-14-SigLIP-384,webli -n $N --resolution 384 384 --patch-size 16 --output-dir $ROOT/open_clip/siglip/384


# python examples/visualize_features.py --dataset $DATASET --model-version InternViT-6B-224px -n $N --resolution 224 224 --output-dir $ROOT/internvit/224
# python examples/visualize_features.py --dataset $DATASET --model-version InternViT-6B-448px-V1-2 -n $N --resolution 448 448 --output-dir $ROOT/internvit/448


# python examples/visualize_features.py --dataset $DATASET --model-version $E_RADIO -n $N --resolution 432 --patch-size 16 --resize-multiple 16 --max-dim --output-dir $ROOT/eradio/432max
# python examples/visualize_features.py --dataset $DATASET --model-version $E_RADIO -n $N --resolution 768 --patch-size 16 --resize-multiple 16 --max-dim --output-dir $ROOT/eradio/768max
# python examples/visualize_features.py --dataset $DATASET --model-version $E_RADIO -n $N --resolution 1024 --patch-size 16 --resize-multiple 16 --max-dim --output-dir $ROOT/eradio/1024max
# python examples/visualize_features.py --dataset $DATASET --model-version $E_RADIO -n $N --resolution 2048 --patch-size 16 --resize-multiple 16 --max-dim --output-dir $ROOT/eradio/2048max

# python im_join.py --input-dirs $ROOT/radiov2/1024min_wnd-16/orig \
#                                $ROOT/open_clip/openai/336/viz \
#                                $ROOT/dinov2_g_reg/518min/viz \
#                                $ROOT/sam_h/viz \
#                                $ROOT/radiov2/512min/viz \
#                                $ROOT/radiov2/1024min/viz \
#                                $ROOT/radiov2/2048max/viz \
#                                $ROOT/radiov2/1024min_wnd-16_sam/viz \
#                                --cols 4 --output-dir cvpr_hires_mosaic

# python im_join.py --input-dirs vis_denoise/radiov2/1024_sam/orig \
#                                vis_denoise/sam_h/viz/backbone \
#                                vis_denoise/radiov2/256_sam/viz/sam \
#                                vis_denoise/radiov2/512_sam/viz/sam \
#                                vis_denoise/radiov2/1024_sam/viz/sam \
#                                vis_denoise/radiov2/2048_sam/viz/sam \
#                   --output-dir sam_mode_switching/radio_v2 --cols 3

# python im_join.py --input-dirs vis_denoise/radiov2/1024_sam/orig \
#                                vis_denoise/sam_h/viz/backbone \
#                                vis_denoise/radiov2.5/vit-h/256_sam/viz/sam \
#                                vis_denoise/radiov2.5/vit-h/512_sam/viz/sam \
#                                vis_denoise/radiov2.5/vit-h/1024_sam/viz/sam \
#                                vis_denoise/radiov2.5/vit-h/2048_sam/viz/sam \
#                   --output-dir sam_mode_switching/radio_v2.5/vit-h --cols 3

# python im_join.py --input-dirs vis_denoise/radiov2/1024_dinov2/orig \
#                                vis_denoise/dinov2_g_reg/224/viz/backbone \
#                                vis_denoise/dinov2_g_reg/518/viz/backbone \
#                                vis_denoise/dinov2_g_reg/1022/viz/backbone \
#                                vis_denoise/radiov2/256_dinov2/viz/dino_v2 \
#                                vis_denoise/radiov2/432_dinov2/viz/dino_v2 \
#                                vis_denoise/radiov2/512_dinov2/viz/dino_v2 \
#                                vis_denoise/radiov2/688_dinov2/viz/dino_v2 \
#                                vis_denoise/radiov2/704_dinov2/viz/dino_v2 \
#                                vis_denoise/radiov2/720_dinov2/viz/dino_v2 \
#                                vis_denoise/radiov2/736_dinov2/viz/dino_v2 \
#                                vis_denoise/radiov2/1024_dinov2/viz/dino_v2 \
#                   --output-dir dinov2_mode_switching/radio_v2 --cols 4

# python im_join.py --input-dirs vis_denoise/radiov2/1024_dinov2/orig \
#                                vis_denoise/dinov2_g_reg/224/viz/backbone \
#                                vis_denoise/dinov2_g_reg/518/viz/backbone \
#                                vis_denoise/dinov2_g_reg/1022/viz/backbone \
#                                vis_denoise/radiov2.5/vit-b/256_dinov2/viz/dino_v2 \
#                                vis_denoise/radiov2.5/vit-b/432_dinov2/viz/dino_v2 \
#                                vis_denoise/radiov2.5/vit-b/512_dinov2/viz/dino_v2 \
#                                vis_denoise/radiov2.5/vit-b/688_dinov2/viz/dino_v2 \
#                                vis_denoise/radiov2.5/vit-b/704_dinov2/viz/dino_v2 \
#                                vis_denoise/radiov2.5/vit-b/720_dinov2/viz/dino_v2 \
#                                vis_denoise/radiov2.5/vit-b/736_dinov2/viz/dino_v2 \
#                                vis_denoise/radiov2.5/vit-b/1024_dinov2/viz/dino_v2 \
#                   --output-dir dinov2_mode_switching/radio_v2.5/vit-b --cols 4

# python im_join.py --input-dirs vis_denoise/radiov2/1024_dinov2/orig \
#                                vis_denoise/dinov2_g_reg/224/viz/backbone \
#                                vis_denoise/dinov2_g_reg/518/viz/backbone \
#                                vis_denoise/dinov2_g_reg/1022/viz/backbone \
#                                vis_denoise/radiov2.5/vit-l/256_dinov2/viz/dino_v2 \
#                                vis_denoise/radiov2.5/vit-l/432_dinov2/viz/dino_v2 \
#                                vis_denoise/radiov2.5/vit-l/512_dinov2/viz/dino_v2 \
#                                vis_denoise/radiov2.5/vit-l/688_dinov2/viz/dino_v2 \
#                                vis_denoise/radiov2.5/vit-l/704_dinov2/viz/dino_v2 \
#                                vis_denoise/radiov2.5/vit-l/720_dinov2/viz/dino_v2 \
#                                vis_denoise/radiov2.5/vit-l/736_dinov2/viz/dino_v2 \
#                                vis_denoise/radiov2.5/vit-l/1024_dinov2/viz/dino_v2 \
#                   --output-dir dinov2_mode_switching/radio_v2.5/vit-l --cols 4

# python im_join.py --input-dirs vis_denoise/radiov2/1024_dinov2/orig \
#                                vis_denoise/dinov2_g_reg/224/viz/backbone \
#                                vis_denoise/dinov2_g_reg/518/viz/backbone \
#                                vis_denoise/dinov2_g_reg/1022/viz/backbone \
#                                vis_denoise/radiov2.5/vit-h/256_dinov2/viz/dino_v2 \
#                                vis_denoise/radiov2.5/vit-h/432_dinov2/viz/dino_v2 \
#                                vis_denoise/radiov2.5/vit-h/512_dinov2/viz/dino_v2 \
#                                vis_denoise/radiov2.5/vit-h/688_dinov2/viz/dino_v2 \
#                                vis_denoise/radiov2.5/vit-h/704_dinov2/viz/dino_v2 \
#                                vis_denoise/radiov2.5/vit-h/720_dinov2/viz/dino_v2 \
#                                vis_denoise/radiov2.5/vit-h/736_dinov2/viz/dino_v2 \
#                                vis_denoise/radiov2.5/vit-h/1024_dinov2/viz/dino_v2 \
#                   --output-dir dinov2_mode_switching/radio_v2.5/vit-h --cols 4

# python im_join.py --input-dirs $ROOT/comm_radio/2048max/orig \
#                                $ROOT/comm_radio/256min/viz \
#                                $ROOT/comm_radio/512min/viz \
#                                $ROOT/comm_radio/768min/viz \
#                                $ROOT/comm_radio/1024min/viz \
#                                $ROOT/comm_radio/2048max/viz \
#                                $ROOT/comm_radio/2048max_clip/viz \
#                                $ROOT/comm_radio/2048max_dino/viz \
#                                $ROOT/comm_radio/2048max_sam/viz \
#                   --output-dir comm_radio_mosaic --cols 3

# python im_join.py --input-dirs $ROOT/radiov2/1024min/orig \
#                                $ROOT/open_clip/dfn5b/378/viz \
#                                $ROOT/dinov2_g_reg/896min/viz \
#                                $ROOT/radiov2/1024min/viz \
#                   --output-dir arxiv_mosaic_dfn-clip

# python im_join.py --input-dirs $ROOT/radiov2/1024min/orig \
#                                $ROOT/openai_clip/vit-l/336/viz \
#                                $ROOT/dinov2_g_reg/896min/viz \
#                                $ROOT/radiov2/1024min/viz \
#                   --output-dir arxiv_mosaic_oai-clip
