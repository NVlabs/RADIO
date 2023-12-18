import os

_base_ = [
    "../_base_/datasets/ade20k.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_80k.py",
]

# model settings
crop_size = (512, 512)
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="RADIO",
        init_cfg=dict(
            type="Pretrained",
        ),
        repo_id="nvidia/E-RADIO",
        token="specify on command line",
    ),
    decode_head=dict(
        type='BNHead',
        in_channels=[1536],
        in_index=[0],
        input_transform='resize_concat',
        channels=1536,
        dropout_ratio=0,
        num_classes=150,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.001, betas=(0.9, 0.999), weight_decay=0.),
    paramwise_cfg=dict(
        custom_keys={
            "pos_embed": dict(decay_mult=0.0),
            "cls_token": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=80000,
        by_epoch=False,
    ),
]

data_root = "/lustre/fs2/portfolios/nvr/projects/nvr_lpr_nvgptvision/datasets/ade20k/ADEChallengeData2016"

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(
    batch_size=2,
    dataset=dict(data_root=data_root))
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="Resize", scale=(2048, 512), keep_ratio=True),
            # add loading annotation after ``Resize`` because ground truth
            # does not need to do resize data transform
            dict(type="LoadAnnotations", reduce_zero_label=True),
            dict(type="PackSegInputs"),
        ]
    ),
)
test_dataloader = val_dataloader

# This is needed to allow distributed training when some parameters
# have no gradient.
find_unused_parameters = True

vis_backends = [
    dict(type="LocalVisBackend"),
]

if "WANDB_API_KEY" in os.environ:
    vis_backends.append(
        dict(
            type="WandbVisBackend",
            init_kwargs=dict(entity="adlr", project="evfm", group="ade20k"),
        )
    )

visualizer = dict(
    type="SegLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)

# Hook for auto-suspend/resume on ADLR clusters.
custom_hooks = [dict(type="AutoResumeHook", interval=8000)]
