#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Generate minimal test dataset for CI testing."""

import json
import os
from pathlib import Path

import cv2
import numpy as np


def create_test_video_metadata(output_dir: Path, num_videos: int = 2, num_frames: int = 10):
    """Create test video files and metadata for training tests.

    Args:
        output_dir: Directory to store metadata and videos
        num_videos: Number of test videos to create
        num_frames: Number of frames per video
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / 'videos'
    videos_dir.mkdir(exist_ok=True)

    print(f"Creating {num_videos} test videos with {num_frames} frames each...")

    for video_idx in range(num_videos):
        video_path = videos_dir / f'test_video_{video_idx:03d}.mp4'

        # Create a simple test video (64x64 resolution for speed)
        height, width = 64, 64
        fps = 30

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        # Generate frames with simple patterns
        for frame_idx in range(num_frames):
            # Create a frame with varying colors (simple test pattern)
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Add some variation per frame
            color_val = int((frame_idx / num_frames) * 255)
            frame[:, :, 0] = color_val  # Blue channel
            frame[:, :, 1] = 255 - color_val  # Green channel
            frame[:, :, 2] = 128  # Red channel (constant)

            # Add a moving rectangle for visual variation
            rect_x = int((frame_idx / num_frames) * (width - 20))
            cv2.rectangle(frame, (rect_x, 20), (rect_x + 20, 44), (255, 255, 255), -1)

            out.write(frame)

        out.release()

        # Create metadata file
        metadata = {
            'video_file': str(video_path.absolute()),
            'video_height': height,
            'video_width': width,
            'video_fps': fps,
            'average_fps': fps,
            'video_fps_exact': f'{fps}/1',
            'codec_name': 'mpeg4',
            'frames_count': num_frames,
            'duration': num_frames / fps,
            'time_base': '1/15360',
            'start_pts': 0,
        }

        metadata_path = output_dir / f'test_video_{video_idx:03d}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  Created: {video_path.name} ({num_frames} frames, {height}x{width})")

    print(f"\nTest data created successfully in: {output_dir}")
    print(f"  - {num_videos} videos")
    print(f"  - {num_videos} metadata files")

    return output_dir


def main():
    """Generate test data for CI."""
    # Create minimal test dataset
    test_data_dir = Path('data/test_ci')

    # Training data
    train_dir = test_data_dir / 'train'
    create_test_video_metadata(train_dir, num_videos=2, num_frames=10)

    # Validation data (optional, can use same as train for testing)
    val_dir = test_data_dir / 'val'
    create_test_video_metadata(val_dir, num_videos=1, num_frames=10)

    print("\n✓ Test dataset generation complete!")
    print(f"  Train: {train_dir}")
    print(f"  Val:   {val_dir}")


if __name__ == '__main__':
    main()
