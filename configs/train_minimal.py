# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Minimal training configuration for quick start.

This config uses minimal settings for faster training and testing.
Good for debugging and quick experiments.
"""

# Model
model = dict(
    type='BasicVSRPlusPlusGan',
    generator=dict(
        type='BasicVSRPlusPlusGanNet',
        mid_channels=64,
        num_blocks=7,  # Reduced from 15 for faster training
        spynet_pretrained=None,
    ),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    is_use_ema=False,  # Disabled for faster training
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ),
)

# Dataset - CHANGE THESE PATHS
train_dataset = dict(
    type='MosaicVideoDataset',
    metadata_root_dir='data/train_metadata',
    lq_size=128,  # Smaller size for faster training
    num_frame=7,  # Fewer frames
    min_num_frame=7,
    use_hflip=True,
    degrade=True,
)

# Dataloader
train_dataloader = dict(
    num_workers=2,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset,
)

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=2e-4, betas=(0.9, 0.999)),
)

# Training
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=10000,  # Short training for testing
    val_interval=1000,
)

# Hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1000,
        by_epoch=False,
        max_keep_ckpts=2,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

# Runtime
default_scope = 'basicvsrpp.mmagic'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=4),
    dist_cfg=dict(backend='nccl'),
)
log_level = 'INFO'
work_dir = './work_dirs/train_minimal'
