# RADIO Semantic Segmentation Linear Probing

This README walks you through the process of training a linear
head on top of the frozen features of a RADIO model in order
to perform semantic segmentation on the ADE20k dataset.

This was tested on a 8xA100 system within a `nvcr.io/nvidia/pytorch:23.11-py3` container.

## Prerequisites

Install required packages:

```Bash
pip install -r requirements.txt
```

## Download ADE20 Dataset

Refer to the instructions on https://groups.csail.mit.edu/vision/datasets/ADE20K/.

## Log In to Hugging Face Hub

```Bash
huggingface-cli login
```

## Train

Specify location of dataset:

```Bash
export ADE20K_ROOT_DIR=/path/to/ade20k/dataset
```

Start training:

```Bash
python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 train.py configs/radio/radio_linear_8xb2-80k_ade20k-512x512.py --launcher pytorch --cfg-options "train_dataloader.dataset.data_root=/${ADE20K_ROOT_DIR}" --cfg-options "val_dataloader.dataset.data_root=${ADE20K_ROOT_DIR}"
```
