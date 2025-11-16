#!/usr/bin/env python3
# Test ModulatedDeformConv2d forward method

import torch
from basicvsrpp.deformconv import ModulatedDeformConv2d

def test_modulated_deform_conv():
    """Test that ModulatedDeformConv2d forward works"""
    print("Testing ModulatedDeformConv2d forward method...")

    # Create a simple instance
    conv = ModulatedDeformConv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        padding=1,
        deform_groups=1
    )
    conv.eval()

    # Create dummy input
    batch_size = 1
    h, w = 32, 32
    x = torch.randn(batch_size, 3, h, w)

    # Create dummy offset and mask
    # offset: (n, deform_groups * 2 * kh * kw, h, w)
    offset = torch.randn(batch_size, 1 * 2 * 3 * 3, h, w)
    # mask: (n, deform_groups * kh * kw, h, w)
    mask = torch.randn(batch_size, 1 * 3 * 3, h, w)

    # Test forward
    with torch.no_grad():
        output = conv(x, offset, mask)

    # Verify output shape
    assert output.shape == (batch_size, 64, h, w), f"Expected shape {(batch_size, 64, h, w)}, got {output.shape}"

    print(f"  ✓ Input shape: {x.shape}")
    print(f"  ✓ Offset shape: {offset.shape}")
    print(f"  ✓ Mask shape: {mask.shape}")
    print(f"  ✓ Output shape: {output.shape}")
    print("  ✓ ModulatedDeformConv2d forward works correctly!")

    return True

if __name__ == "__main__":
    try:
        test_modulated_deform_conv()
        print("\n✓ All tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
