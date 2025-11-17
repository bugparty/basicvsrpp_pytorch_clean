#!/usr/bin/env python3
"""Test inference functionality for CI."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from basicvsrpp.inference import load_model, inference, get_default_gan_inference_config
from basicvsrpp import register_all_modules
from basicvsrpp.mmagic.registry import MODELS
import torch
import numpy as np


def test_inference_config():
    """Test that inference config can be created."""
    print("Testing inference config creation...")
    register_all_modules()
    config = get_default_gan_inference_config()
    print(f"✓ Inference config created successfully")
    print(f"  Model type: {config['type']}")
    assert config['type'] == 'BasicVSRPlusPlusGan'


def test_model_building():
    """Test that model can be built from config."""
    print("\nTesting model building...")
    register_all_modules()
    config = get_default_gan_inference_config()
    model = MODELS.build(config)
    model.eval()
    print(f"✓ Model built successfully: {type(model).__name__}")


if __name__ == '__main__':
    try:
        test_inference_config()
        test_model_building()
        print("\n✓ All inference tests passed!")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Inference tests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
