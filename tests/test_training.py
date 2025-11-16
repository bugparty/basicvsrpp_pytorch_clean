#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Test training functionality."""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from mmengine.config import Config
from mmengine.runner import Runner

from basicvsrpp import register_all_modules


def test_config_loading():
    """Test that training config can be loaded."""
    print("Testing config loading...")

    cfg = Config.fromfile('configs/test_ci.py')

    assert cfg.model.type == 'BasicVSRPlusPlusGan'
    assert cfg.train_cfg.max_iters == 10
    print("  ✓ Config loaded successfully")


def test_model_building():
    """Test that model can be built from config."""
    print("Testing model building...")

    register_all_modules()
    cfg = Config.fromfile('configs/test_ci.py')

    from basicvsrpp.mmagic.registry import MODELS
    model = MODELS.build(cfg.model)
    model.eval()

    print(f"  ✓ Model built: {type(model).__name__}")


def test_runner_creation():
    """Test that runner can be created."""
    print("Testing runner creation...")

    register_all_modules()
    cfg = Config.fromfile('configs/test_ci.py')

    # Use temporary work directory
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.work_dir = tmpdir
        runner = Runner.from_cfg(cfg)
        print(f"  ✓ Runner created successfully")
        print(f"    Work dir: {tmpdir}")


def test_single_iteration():
    """Test running a single training iteration."""
    print("Testing single training iteration...")

    # Check if test data exists
    test_data_dir = Path('data/test_ci/train')
    if not test_data_dir.exists():
        print("  ⚠ Test data not found. Run 'python tests/generate_test_data.py' first.")
        print("  Skipping single iteration test.")
        return

    register_all_modules()
    cfg = Config.fromfile('configs/test_ci.py')

    # Override to just 1 iteration for quick test
    cfg.train_cfg.max_iters = 1

    # Use temporary work directory
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.work_dir = tmpdir
        runner = Runner.from_cfg(cfg)

        # Run training
        try:
            runner.train()
            print("  ✓ Single iteration completed successfully")

            # Check that checkpoint was created
            ckpt_path = Path(tmpdir) / 'iter_1.pth'
            if ckpt_path.exists():
                print(f"  ✓ Checkpoint created: {ckpt_path.name}")
            else:
                print("  ⚠ Checkpoint not found (might be expected if interval > 1)")
        except Exception as e:
            print(f"  ✗ Training failed: {e}")
            raise


def main():
    """Run all training tests."""
    print("="*60)
    print("Training Functionality Tests")
    print("="*60)

    try:
        test_config_loading()
        test_model_building()
        test_runner_creation()
        test_single_iteration()

        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        return 0

    except Exception as e:
        print("\n" + "="*60)
        print(f"✗ Tests failed: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
