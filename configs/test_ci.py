# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Minimal configuration for CI testing.

This config is designed to run quickly on CPU for testing purposes.
Not suitable for actual training.
"""

# Model - minimal version for CPU testing
model = dict(
    type='BasicVSRPlusPlusGan',
    generator=dict(
        type='BasicVSRPlusPlusGanNet',
        mid_channels=8,  # Very small for CPU
        num_blocks=1,  # Minimal blocks
        spynet_pretrained=None,
    ),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    is_use_ema=False,  # Disabled for faster testing
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ),
)

# Dataset - minimal test data
train_dataset = dict(
    type='MosaicVideoDataset',
    metadata_root_dir='data/test_ci/train',
    lq_size=64,  # Small size for CPU
    num_frame=5,  # Few frames
    min_num_frame=5,
    use_hflip=False,  # Disable augmentation for speed
    degrade=False,  # Disable degradation for speed
    random_mosaic_params=False,
    repeatable_random=True,
)

# Dataloader - minimal
train_dataloader = dict(
    num_workers=0,  # No multiprocessing on CI
    batch_size=1,  # Minimal batch size
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=train_dataset,
)

# Optimizer - simple
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-3, betas=(0.9, 0.999)),
)

# Training - very short
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=10,  # Just 10 iterations for testing
    val_interval=1000,  # No validation during test
)

# Hooks - minimal logging
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),  # Log every iteration
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,  # Save every 5 iterations
        by_epoch=False,
        max_keep_ckpts=2,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

# Runtime
default_scope = 'basicvsrpp.mmagic'
env_cfg = dict(
    cudnn_benchmark=False,  # Disabled for CPU
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=1),
    dist_cfg=dict(backend='gloo'),  # Use gloo for CPU
)
log_level = 'INFO'
work_dir = './work_dirs/test_ci'
