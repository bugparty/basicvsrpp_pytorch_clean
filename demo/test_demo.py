#!/usr/bin/env python3
"""
测试 demo 脚本的基本功能（不需要实际的 checkpoint）
"""

import sys
import os
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2


def create_test_video(output_path, num_frames=10, fps=25, size=(256, 256)):
    """创建测试视频"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, size)

    for i in range(num_frames):
        # 创建渐变的测试帧
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        frame[:, :] = (i * 25, 128, 255 - i * 25)
        video_writer.write(frame)

    video_writer.release()
    print(f"Created test video: {output_path}")


def create_test_frames(output_dir, num_frames=10, size=(256, 256)):
    """创建测试帧序列"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_frames):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        frame[:, :] = (i * 25, 128, 255 - i * 25)
        cv2.imwrite(str(output_dir / f"{i:08d}.png"), frame)

    print(f"Created {num_frames} test frames in: {output_dir}")


def test_video_io():
    """测试视频读写功能"""
    print("\n=== Testing Video I/O ===")

    from demo.restoration_video_demo import read_video_frames, write_video_frames

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建测试视频
        input_video = os.path.join(tmpdir, "input.mp4")
        output_video = os.path.join(tmpdir, "output.mp4")
        create_test_video(input_video, num_frames=10)

        # 读取视频
        frames, fps = read_video_frames(input_video)
        print(f"✓ Read {len(frames)} frames (fps: {fps})")
        assert len(frames) == 10, f"Expected 10 frames, got {len(frames)}"

        # 写入视频
        write_video_frames(frames, output_video, fps=fps)
        print(f"✓ Wrote video to {output_video}")

        # 验证输出
        verify_frames, verify_fps = read_video_frames(output_video)
        print(f"✓ Verified {len(verify_frames)} frames in output")
        assert len(verify_frames) == len(frames), "Frame count mismatch"


def test_image_io():
    """测试图像序列读写功能"""
    print("\n=== Testing Image Sequence I/O ===")

    from demo.restoration_video_demo import read_image_frames, write_image_frames

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建测试帧
        input_dir = os.path.join(tmpdir, "input_frames")
        output_dir = os.path.join(tmpdir, "output_frames")
        create_test_frames(input_dir, num_frames=10)

        # 读取帧
        frames = read_image_frames(input_dir)
        print(f"✓ Read {len(frames)} frames from directory")
        assert len(frames) == 10, f"Expected 10 frames, got {len(frames)}"

        # 写入帧
        write_image_frames(frames, output_dir)
        print(f"✓ Wrote frames to {output_dir}")

        # 验证输出
        verify_frames = read_image_frames(output_dir)
        print(f"✓ Verified {len(verify_frames)} frames in output")
        assert len(verify_frames) == len(frames), "Frame count mismatch"


def test_imports():
    """测试 demo 的导入"""
    print("\n=== Testing Demo Imports ===")

    try:
        from demo.restoration_video_demo import (
            read_video_frames,
            read_image_frames,
            write_video_frames,
            write_image_frames,
            parse_args
        )
        print("✓ All demo functions imported successfully")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        raise


def main():
    """运行所有测试"""
    print("Testing BasicVSR++ Demo Functions\n")

    try:
        test_imports()
        test_video_io()
        test_image_io()

        print("\n" + "=" * 50)
        print("✓ All tests passed!")
        print("=" * 50)
        return 0

    except Exception as e:
        print("\n" + "=" * 50)
        print(f"✗ Tests failed: {e}")
        print("=" * 50)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
