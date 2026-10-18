#!/usr/bin/env python3
# Test basic functionality

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

def test_registry():
    """Test that the registry system works"""
    print("Testing registry system...")

    from basicvsrpp.mmagic.registry import MODELS
    from basicvsrpp import register_all_modules

    # Register all modules
    register_all_modules()

    # Check that models are registered
    assert 'BasicVSRPlusPlusGan' in MODELS.module_dict, "BasicVSRPlusPlusGan not registered"
    assert 'BasicVSRPlusPlusGanNet' in MODELS.module_dict, "BasicVSRPlusPlusGanNet not registered"

    print("  ✓ Registry system works")

def test_config():
    """Test that the default config can be created"""
    print("Testing config creation...")

    from basicvsrpp.inference import get_default_gan_inference_config

    config = get_default_gan_inference_config()

    assert config['type'] == 'BasicVSRPlusPlusGan', "Config type incorrect"
    assert 'generator' in config, "Generator not in config"
    assert config['generator']['type'] == 'BasicVSRPlusPlusGanNet', "Generator type incorrect"

    print("  ✓ Config creation works")

def test_model_build():
    """Test that the model can be built from config"""
    print("Testing model building...")

    from basicvsrpp.mmagic.registry import MODELS
    from basicvsrpp import register_all_modules
    from basicvsrpp.inference import get_default_gan_inference_config

    # Register all modules
    register_all_modules()

    # Get config
    config = get_default_gan_inference_config()

    # Build model
    try:
        model = MODELS.build(config)
        print("  ✓ Model built successfully")

        # Check model is in eval mode after calling eval()
        model.eval()
        assert not model.training, "Model should be in eval mode"
        print("  ✓ Model can switch to eval mode")

    except Exception as e:
        print(f"  ✗ Model building failed: {e}")
        raise

def test_image_utils():
    """Test image utility functions"""
    print("Testing image utilities...")

    from lib.image_utils import img2tensor, tensor2img

    # Create a dummy image
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Test img2tensor
    tensor = img2tensor([img], bgr2rgb=False, float32=True)
    assert len(tensor) == 1, "Should return list of tensors"
    assert isinstance(tensor[0], torch.Tensor), "Should return torch.Tensor"

    # Test tensor2img
    imgs_back = tensor2img(tensor, rgb2bgr=False, out_type=np.uint8, min_max=(0, 1))
    assert len(imgs_back) == 1, "Should return list of images"
    assert isinstance(imgs_back[0], np.ndarray), "Should return numpy array"

    print("  ✓ Image utilities work")

def test_deformable_conv():
    """Test that deformable convolution can be imported"""
    print("Testing deformable convolution...")

    from basicvsrpp.deformconv import ModulatedDeformConv2d

    # Create a small instance (don't need to run it, just check it can be created)
    conv = ModulatedDeformConv2d(3, 64, kernel_size=3, padding=1)

    print("  ✓ Deformable convolution can be instantiated")

def main():
    """Run all tests"""
    print("Running basic functionality tests...\n")

    try:
        test_registry()
        test_config()
        test_model_build()
        test_image_utils()
        test_deformable_conv()

        print("\n✓ All tests passed!")
        return 0
    except Exception as e:
        print(f"\n✗ Tests failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
