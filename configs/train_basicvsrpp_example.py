# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Example training configuration for BasicVSR++ GAN.

This is a minimal example configuration. Adjust the parameters according to your needs.
"""

# =============================================================================
# Model Configuration
# =============================================================================
model = dict(
    type='BasicVSRPlusPlusGan',
    generator=dict(
        type='BasicVSRPlusPlusGanNet',
        mid_channels=64,
        num_blocks=15,
        spynet_pretrained=None,  # Path to pretrained SPyNet weights (optional)
    ),
    discriminator=dict(
        type='UNetDiscriminatorWithSpectralNorm',
        in_channels=3,
        mid_channels=64,
        skip_connection=True,
    ),
    pixel_loss=dict(
        type='CharbonnierLoss',
        loss_weight=1.0,
        reduction='mean',
    ),
    perceptual_loss=dict(
        type='PerceptualLoss',
        layer_weights={
            '2': 0.1,
            '7': 0.1,
            '16': 1.0,
            '25': 1.0,
            '34': 1.0,
        },
        vgg_type='vgg19',
        perceptual_weight=1.0,
        style_weight=0,
        criterion='l1',
    ),
    gan_loss=dict(
        type='GANLoss',
        gan_type='vanilla',
        loss_weight=0.1,
        real_label_val=1.0,
        fake_label_val=0,
    ),
    is_use_ema=True,
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ),
    train_cfg=dict(
        disc_steps=1,
        disc_init_steps=0,
    ),
)

# =============================================================================
# Dataset Configuration
# =============================================================================
train_dataset = dict(
    type='MosaicVideoDataset',
    metadata_root_dir='data/train_metadata',  # CHANGE THIS to your metadata directory
    scale=1,
    lq_size=256,
    num_frame=15,
    min_num_frame=15,
    use_hflip=True,
    degrade=True,
    random_mosaic_params=True,
    repeatable_random=False,
)

val_dataset = dict(
    type='MosaicVideoDataset',
    metadata_root_dir='data/val_metadata',  # CHANGE THIS to your metadata directory
    scale=1,
    lq_size=256,
    num_frame=15,
    min_num_frame=15,
    use_hflip=False,
    degrade=False,
    random_mosaic_params=False,
    repeatable_random=True,
)

# =============================================================================
# Dataloader Configuration
# =============================================================================
train_dataloader = dict(
    num_workers=4,
    batch_size=2,  # Batch size per GPU
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset,
)

val_dataloader = dict(
    num_workers=4,
    batch_size=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset,
)

# =============================================================================
# Optimizer Configuration
# =============================================================================
optim_wrapper = dict(
    constructor='MultiOptimWrapperConstructor',
    generator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)),
    ),
    discriminator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)),
    ),
)

# =============================================================================
# Learning Rate Scheduler
# =============================================================================
param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=False,
    T_max=600000,
    eta_min=1e-7,
)

# =============================================================================
# Training Configuration
# =============================================================================
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=600000,
    val_interval=5000,
)

val_cfg = dict(type='MultiValLoop')

# =============================================================================
# Evaluation Configuration
# =============================================================================
val_evaluator = dict(
    type='Evaluator',
    metrics=[
        dict(type='PSNR', crop_border=0),
        dict(type='SSIM', crop_border=0),
    ],
)

# =============================================================================
# Hooks Configuration
# =============================================================================
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5000,
        by_epoch=False,
        max_keep_ckpts=3,
        save_best='PSNR',
        rule='greater',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

custom_hooks = [
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema', ),
        interval=1,
        start_iter=0,
        momentum=0.999,
    ),
    dict(
        type='VisualizationHook',
        interval=5000,
        res_name_list=['gt_img', 'input', 'output'],
    ),
]

# =============================================================================
# Runtime Configuration
# =============================================================================
default_scope = 'basicvsrpp.mmagic'

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=4),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(
    type='LogProcessor',
    window_size=100,
    by_epoch=False,
    custom_cfg=None,
    num_digits=4,
)

log_level = 'INFO'

load_from = None  # Path to pretrained checkpoint (optional)
resume = False

# Visualization backend
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ConcatImageVisualizer',
    vis_backends=vis_backends,
    fn_key='gt_path',
    img_keys=['gt_img', 'input', 'output'],
    bgr2rgb=True,
)

# Working directory
work_dir = './work_dirs/train_basicvsrpp'

# Base batch size for auto-scaling learning rate
base_batch_size = 8  # 4 GPUs x 2 samples per GPU
