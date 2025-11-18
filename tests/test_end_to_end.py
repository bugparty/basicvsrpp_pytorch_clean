#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""End-to-end inference tests for BasicVSR++"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from basicvsrpp import register_all_modules
from basicvsrpp.mmagic.registry import MODELS
from basicvsrpp.inference import get_default_gan_inference_config, inference
from lib.image_utils import img2tensor, tensor2img


def test_model_forward():
    """Test that model forward pass works with synthetic data"""
    print("Testing model forward pass...")

    register_all_modules()
    config = get_default_gan_inference_config()

    # Build model
    model = MODELS.build(config)
    model.eval()

    # Create synthetic input (B, T, C, H, W)
    # NOTE: Input must be at least 256x256 because model downsamples by 4x
    # and requires downsampled size to be at least 64x64
    batch_size = 1
    num_frames = 4
    height, width = 256, 256

    # Create random input tensor
    input_tensor = torch.randn(batch_size, num_frames, 3, height, width)

    # Run forward pass
    with torch.no_grad():
        output = model(inputs=input_tensor)

    # Check output shape (should be 4x upsampled)
    expected_shape = (batch_size, num_frames, 3, height * 4, width * 4)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

    # Check output is not all zeros or NaN
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"

    print(f"  ✓ Model forward pass works (output shape: {output.shape})")


def test_inference_with_numpy_images():
    """Test inference function with numpy array images"""
    print("Testing inference with numpy images...")

    register_all_modules()
    config = get_default_gan_inference_config()

    # Build model
    model = MODELS.build(config)
    model.eval()

    # Create synthetic video frames (list of numpy arrays)
    # NOTE: Input must be at least 256x256
    num_frames = 4
    height, width = 256, 256
    video = [
        np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        for _ in range(num_frames)
    ]

    # Run inference
    device = torch.device('cpu')
    output = inference(model, video, device)

    # Check output
    assert len(output) == num_frames, f"Expected {num_frames} frames, got {len(output)}"
    assert output[0].shape == (height * 4, width * 4, 3), \
        f"Expected shape {(height * 4, width * 4, 3)}, got {output[0].shape}"
    assert output[0].dtype == np.uint8, f"Expected dtype uint8, got {output[0].dtype}"

    print(f"  ✓ Inference with numpy images works ({len(output)} frames processed)")


def test_inference_with_batching():
    """Test inference with batching (max_frames parameter)"""
    print("Testing inference with batching...")

    register_all_modules()
    config = get_default_gan_inference_config()

    # Build model
    model = MODELS.build(config)
    model.eval()

    # Create synthetic video with more frames
    # NOTE: Input must be at least 256x256
    num_frames = 10
    height, width = 256, 256
    video = [
        np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        for _ in range(num_frames)
    ]

    # Run inference with batching (process 3 frames at a time)
    device = torch.device('cpu')
    output = inference(model, video, device, max_frames=3)

    # Check output
    assert len(output) == num_frames, f"Expected {num_frames} frames, got {len(output)}"
    assert output[0].shape == (height * 4, width * 4, 3), \
        f"Expected shape {(height * 4, width * 4, 3)}, got {output[0].shape}"

    print(f"  ✓ Batched inference works ({len(output)} frames processed in batches of 3)")


def test_inference_consistency():
    """Test that inference produces consistent results for the same input"""
    print("Testing inference consistency...")

    register_all_modules()
    config = get_default_gan_inference_config()

    # Build model
    model = MODELS.build(config)
    model.eval()

    # Create synthetic video
    # NOTE: Input must be at least 256x256
    num_frames = 3
    height, width = 256, 256
    video = [
        np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        for _ in range(num_frames)
    ]

    device = torch.device('cpu')

    # Run inference twice
    output1 = inference(model, video, device)
    output2 = inference(model, video, device)

    # Check that outputs are identical
    for i, (frame1, frame2) in enumerate(zip(output1, output2)):
        assert np.array_equal(frame1, frame2), \
            f"Frame {i} differs between runs (should be deterministic)"

    print("  ✓ Inference produces consistent results")


def test_different_input_sizes():
    """Test inference with different input resolutions"""
    print("Testing different input sizes...")

    register_all_modules()
    config = get_default_gan_inference_config()

    # Build model
    model = MODELS.build(config)
    model.eval()

    device = torch.device('cpu')

    # Test different sizes (all must be at least 256x256)
    test_sizes = [(256, 256), (256, 320), (320, 256)]

    for height, width in test_sizes:
        video = [
            np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            for _ in range(3)
        ]

        output = inference(model, video, device)

        assert len(output) == 3, f"Expected 3 frames for size {height}x{width}"
        assert output[0].shape == (height * 4, width * 4, 3), \
            f"Unexpected output shape for input {height}x{width}"

    print(f"  ✓ Inference works with different input sizes ({len(test_sizes)} sizes tested)")


def test_spynet_forward():
    """Test SPyNet optical flow estimation"""
    print("Testing SPyNet forward pass...")

    from basicvsrpp.mmagic.basicvsr_plusplus_net import SPyNet

    # Create SPyNet instance
    spynet = SPyNet(pretrained=None)
    spynet.eval()

    # Create synthetic image pair
    batch_size = 1
    height, width = 128, 128
    ref = torch.randn(batch_size, 3, height, width)
    supp = torch.randn(batch_size, 3, height, width)

    # Compute optical flow
    with torch.no_grad():
        flow = spynet(ref, supp)

    # Check output shape (flow should have 2 channels for x and y)
    expected_shape = (batch_size, 2, height, width)
    assert flow.shape == expected_shape, f"Expected shape {expected_shape}, got {flow.shape}"

    # Check flow is not all zeros or NaN
    assert not torch.isnan(flow).any(), "Flow contains NaN values"
    assert not torch.isinf(flow).any(), "Flow contains Inf values"

    print(f"  ✓ SPyNet forward pass works (flow shape: {flow.shape})")


def test_second_order_deformable_alignment():
    """Test SecondOrderDeformableAlignment module"""
    print("Testing SecondOrderDeformableAlignment...")

    from basicvsrpp.mmagic.basicvsr_plusplus_net import SecondOrderDeformableAlignment

    # Create alignment module
    alignment = SecondOrderDeformableAlignment(
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        padding=1,
        deform_groups=16
    )
    alignment.eval()

    # Create synthetic inputs
    batch_size = 1
    height, width = 64, 64
    channels = 64

    x = torch.randn(batch_size, channels, height, width)
    extra_feat = torch.randn(batch_size, channels, height, width)
    flow_1 = torch.randn(batch_size, 2, height, width)
    flow_2 = torch.randn(batch_size, 2, height, width)

    # Run forward pass
    with torch.no_grad():
        output = alignment(x, extra_feat, flow_1, flow_2)

    # Check output shape
    expected_shape = (batch_size, channels, height, width)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

    # Check output is valid
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"

    print(f"  ✓ SecondOrderDeformableAlignment works (output shape: {output.shape})")


def test_basicvsr_plusplus_net():
    """Test BasicVSRPlusPlusNet end-to-end"""
    print("Testing BasicVSRPlusPlusNet...")

    from basicvsrpp.mmagic.basicvsr_plusplus_net import BasicVSRPlusPlusNet

    # Create network
    net = BasicVSRPlusPlusNet(
        mid_channels=64,
        num_blocks=7,  # Use fewer blocks for faster testing
        spynet_pretrained=None
    )
    net.eval()

    # Create synthetic input (B, T, C, H, W)
    # NOTE: Input must be at least 256x256
    batch_size = 1
    num_frames = 5
    height, width = 256, 256

    lqs = torch.randn(batch_size, num_frames, 3, height, width)

    # Run forward pass
    with torch.no_grad():
        output = net(lqs)

    # Check output shape (4x upsampled)
    expected_shape = (batch_size, num_frames, 3, height * 4, width * 4)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

    # Check output is valid
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"

    print(f"  ✓ BasicVSRPlusPlusNet works (output shape: {output.shape})")


def main():
    """Run all end-to-end tests"""
    print("Running end-to-end tests...\n")

    try:
        # Core inference tests
        test_model_forward()
        test_inference_with_numpy_images()
        test_inference_with_batching()
        test_inference_consistency()
        test_different_input_sizes()

        # Component tests
        test_spynet_forward()
        test_second_order_deformable_alignment()
        test_basicvsr_plusplus_net()

        print("\n✓ All end-to-end tests passed!")
        return 0
    except Exception as e:
        print(f"\n✗ Tests failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
