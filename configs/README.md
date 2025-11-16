# Training Configurations

This directory contains training configuration files for BasicVSR++.

## Available Configs

- **`train_basicvsrpp_example.py`**: Full-featured training configuration with GAN, perceptual loss, and all bells and whistles. Good starting point for serious training.
- **`train_minimal.py`**: Minimal configuration for quick testing and debugging. Uses smaller models and fewer iterations.

## Configuration Structure

A training config file typically contains the following sections:

### 1. Model Configuration
```python
model = dict(
    type='BasicVSRPlusPlusGan',
    generator=dict(...),
    discriminator=dict(...),  # Optional, for GAN training
    pixel_loss=dict(...),
    perceptual_loss=dict(...),  # Optional
    gan_loss=dict(...),  # Optional
    # ...
)
```

### 2. Dataset Configuration
```python
train_dataset = dict(
    type='MosaicVideoDataset',
    metadata_root_dir='path/to/metadata',
    lq_size=256,
    num_frame=15,
    # ...
)
```

### 3. Dataloader Configuration
```python
train_dataloader = dict(
    num_workers=4,
    batch_size=2,
    dataset=train_dataset,
    # ...
)
```

### 4. Optimizer Configuration
```python
optim_wrapper = dict(
    # For single optimizer
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-4),
)

# Or for GAN training with multiple optimizers
optim_wrapper = dict(
    constructor='MultiOptimWrapperConstructor',
    generator=dict(...),
    discriminator=dict(...),
)
```

### 5. Training Configuration
```python
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=600000,
    val_interval=5000,
)
```

## Data Preparation

Before training, you need to prepare your dataset metadata. The `MosaicVideoDataset` expects metadata files that describe your video dataset.

### Metadata Structure

Create a metadata directory with the following structure:
```
data/
├── train_metadata/
│   ├── video1_metadata.json
│   ├── video2_metadata.json
│   └── ...
└── val_metadata/
    ├── video1_metadata.json
    └── ...
```

Each metadata file should contain information about a video file. See `lib/restoration_dataset_metadata.py` for the metadata format.

## Usage

### Single GPU Training
```bash
python tools/train.py configs/train_minimal.py
```

### Multi-GPU Training
```bash
# Using torchrun (recommended)
torchrun --nproc_per_node=4 tools/train.py configs/train_basicvsrpp_example.py

# Or using torch.distributed.launch (older)
python -m torch.distributed.launch --nproc_per_node=4 tools/train.py configs/train_basicvsrpp_example.py
```

### Resume Training
```bash
python tools/train.py configs/train_minimal.py --resume work_dirs/train_minimal/latest.pth
```

### Override Config Options
```bash
python tools/train.py configs/train_minimal.py \
    --cfg-options train_cfg.max_iters=20000 \
    optim_wrapper.optimizer.lr=2e-4
```

## Key Parameters to Adjust

| Parameter | Location | Description |
|-----------|----------|-------------|
| `metadata_root_dir` | Dataset config | Path to your dataset metadata |
| `batch_size` | Dataloader config | Batch size per GPU |
| `num_workers` | Dataloader config | Number of data loading workers |
| `lr` | Optimizer config | Learning rate |
| `max_iters` | Training config | Total training iterations |
| `mid_channels` | Model config | Model capacity (64 for standard) |
| `num_blocks` | Model config | Model depth (15 for full, 7 for fast) |

## Tips

1. **Start Small**: Use `train_minimal.py` first to verify your setup works
2. **Monitor GPU Memory**: Reduce `batch_size` or `lq_size` if you run out of memory
3. **Adjust num_workers**: Set to `num_workers = num_cpus / num_gpus` for best performance
4. **Use AMP for Speed**: Add `--amp` flag to enable automatic mixed precision training
5. **Check Logs**: Training logs and checkpoints are saved in `work_dir`

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Reduce `lq_size` (image size)
- Reduce `num_frame` (number of frames)
- Reduce `mid_channels` or `num_blocks`

### Slow Training
- Increase `num_workers` for faster data loading
- Use `--amp` for mixed precision training
- Increase `batch_size` if you have GPU memory
- Use multiple GPUs

### Dataset Issues
- Verify `metadata_root_dir` path is correct
- Check metadata files are valid JSON
- Ensure video files referenced in metadata exist
