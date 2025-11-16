# Testing Guide

Quick guide for testing the training code locally before pushing to GitHub.

## Quick Test (5 minutes)

```bash
# 1. Generate test data
python tests/generate_test_data.py

# 2. Run training tests
python tests/test_training.py

# 3. Run actual training (2 iterations)
python tools/train.py configs/test_ci.py \
  --cfg-options train_cfg.max_iters=2 \
  default_hooks.checkpoint.interval=1
```

## What Gets Tested

The tests verify:
- ✅ All modules can be imported
- ✅ Model can be built from config
- ✅ Training runs without errors
- ✅ Checkpoints are saved
- ✅ Training can be resumed

## Test Data

Test data is synthetic and minimal:
- 2-3 tiny videos (64x64 resolution)
- 10 frames each
- ~1MB total size
- Generated automatically by script

## CI Tests

When you push, GitHub Actions will automatically:
1. Install dependencies (PyTorch CPU version)
2. Generate test data
3. Run 2 training iterations
4. Test checkpoint saving
5. Test resume training
6. Run unit tests

View results at: `https://github.com/your-repo/actions`

## Troubleshooting

**"Test data not found"**
```bash
python tests/generate_test_data.py
```

**"Module not found"**
```bash
pip install -r requirements.txt
```

**"CUDA out of memory" (on CI)**
- CI uses CPU only, this shouldn't happen
- If it does, reduce model size in `configs/test_ci.py`

## Manual Testing

To test the full training pipeline:

```bash
# Small test (10 iterations, ~1 minute)
python tools/train.py configs/test_ci.py

# Minimal real training (1000 iterations)
python tools/train.py configs/train_minimal.py \
  --cfg-options train_cfg.max_iters=1000

# Check outputs
ls work_dirs/train_minimal/
```
