#!/usr/bin/env python3
# Test that all main modules can be imported

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all main modules can be imported"""
    print("Testing imports...")

    # Test basicvsrpp imports
    print("  - Testing basicvsrpp imports...")
    try:
        import basicvsrpp
        print("    ✓ basicvsrpp")
    except Exception as e:
        print(f"    ✗ basicvsrpp: {e}")
        raise

    try:
        from basicvsrpp import register_all_modules
        print("    ✓ basicvsrpp.register_all_modules")
    except Exception as e:
        print(f"    ✗ basicvsrpp.register_all_modules: {e}")
        raise

    # Test mmagic imports
    print("  - Testing mmagic imports...")
    try:
        from basicvsrpp.mmagic import registry
        print("    ✓ basicvsrpp.mmagic.registry")
    except Exception as e:
        print(f"    ✗ basicvsrpp.mmagic.registry: {e}")
        raise

    try:
        from basicvsrpp.mmagic.basicvsr_plusplus_net import BasicVSRPlusPlusNet
        print("    ✓ basicvsrpp.mmagic.basicvsr_plusplus_net")
    except Exception as e:
        print(f"    ✗ basicvsrpp.mmagic.basicvsr_plusplus_net: {e}")
        raise

    # Test lib imports
    print("  - Testing lib imports...")
    try:
        import lib
        print("    ✓ lib")
    except Exception as e:
        print(f"    ✗ lib: {e}")
        raise

    try:
        from lib.image_utils import img2tensor, tensor2img
        print("    ✓ lib.image_utils")
    except Exception as e:
        print(f"    ✗ lib.image_utils: {e}")
        raise

    print("\n✓ All imports successful!")

if __name__ == "__main__":
    test_imports()
