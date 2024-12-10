#!/bin/bash
set -o xtrace

ROOT="vis_denoise"
N="128"

DATASET="/lustre/fsw/portfolios/llmservice/users/mranzinger/data/radio/paper_images"
RADIOV2="radio_v2.1"

export PYTHONPATH=.:examples

# Some models:
# OpenAI CLIP:  openai_clip,ViT-L/14@336px
# DFN CLIP:     open_clip,ViT-H-14-378-quickgelu,dfn5b
# DINOv2-g-reg: dinov2_vitg14_reg

exp="xpos"
fspath="/lustre/fsw/portfolios/llmservice/users/mranzinger/output/evfm/$exp"
resolutions=(256 512 768 1024 2048)
adaptors=("" "clip" "paligemma-448" "sam")
for adaptor in "${adaptors[@]}";
do
    AD_FLAG=""
    AD_SUFF=""
    VIZ_PATH="backbone"
    if [[ -n "$adaptor" ]]; then
        AD_FLAG="--adaptor-name $adaptor"
        AD_SUFF="_${adaptor}"
        VIZ_PATH="$adaptor"
    fi

    JOINED=()
    for modname in `ls "$fspath"`;
    do
        chk="${fspath}/${modname}/checkpoints/last_release_half.pth.tar"

        OUTDIRS=("$ROOT/$exp/${modname}/512min${AD_SUFF}/orig")
        for res in "${resolutions[@]}";
        do
            outdir="$ROOT/$exp/${modname}/${res}min${AD_SUFF}"

            OUTDIRS+=("${outdir}/viz/${VIZ_PATH}")
            if [[ ! -d "$outdir" ]]; then
                python examples/visualize_features.py --dataset $DATASET --model-version "$chk" -n $N --resolution $res $AD_FLAG --output-dir "$outdir" --interpolation nearest
            fi
        done

        join_path="$ROOT/$exp/${modname}/join${AD_SUFF}"
        if [[ ! -d "$join_path" ]]; then
            header_txt="0,0,${modname}"
            col=1
            for res in "${resolutions[@]}";
            do
                header_txt="${header_txt};0,$col,$res"
                col+=1
            done

            python tools/im_join.py --input-dirs ${OUTDIRS[@]} --output-dir "$join_path" --cols 6 --header 64 --header-text "$header_txt"
        fi
        if [[ -d "$join_path" ]]; then
            JOINED+=("$join_path")
        fi
    done

    python tools/im_join.py --input-dirs ${JOINED[@]} --output-dir "$ROOT/$exp/join${AD_SUFF}" --cols 1 --header 0
done
