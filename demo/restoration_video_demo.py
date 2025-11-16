#!/usr/bin/env python3
# Adapted from: https://github.com/ckkelvinchan/BasicVSR_PlusPlus/blob/master/demo/restoration_video_demo.py
# Modified to work with the clean BasicVSR++ implementation

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from basicvsrpp.inference import load_model, inference, get_default_gan_inference_config

VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv')


def parse_args():
    parser = argparse.ArgumentParser(description='BasicVSR++ Video Restoration Demo')
    parser.add_argument('input', help='input video file or directory of frames')
    parser.add_argument('output', help='output video file or directory for frames')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument(
        '--config',
        default=None,
        help='config file path (optional, uses default if not provided)')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='device to use (e.g., cuda:0, cpu)')
    parser.add_argument(
        '--max-frames',
        type=int,
        default=-1,
        help='maximum frames to process at once (-1 for all)')
    parser.add_argument(
        '--fps',
        type=float,
        default=25.0,
        help='fps for output video (default: 25)')
    args = parser.parse_args()
    return args


def read_video_frames(video_path):
    """Read frames from video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames, fps


def read_image_frames(image_dir):
    """Read frames from image directory."""
    image_dir = Path(image_dir)
    image_files = sorted(image_dir.glob('*.png')) + sorted(image_dir.glob('*.jpg'))

    if not image_files:
        raise ValueError(f"No image files found in {image_dir}")

    frames = []
    for img_path in image_files:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Warning: Could not read {img_path}")
            continue
        frames.append(frame)

    return frames


def write_video_frames(frames, output_path, fps=25.0):
    """Write frames to video file."""
    if not frames:
        raise ValueError("No frames to write")

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames:
        video_writer.write(frame.astype(np.uint8))

    video_writer.release()
    print(f"Video saved to: {output_path}")


def write_image_frames(frames, output_dir):
    """Write frames to image directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        output_path = output_dir / f"{i:08d}.png"
        cv2.imwrite(str(output_path), frame)

    print(f"Frames saved to: {output_dir}")


def main():
    """Demo for BasicVSR++ video restoration.

    Examples:
        # Video input/output
        python demo/restoration_video_demo.py input.mp4 output.mp4 checkpoint.pth

        # Image sequence input/output
        python demo/restoration_video_demo.py input_frames/ output_frames/ checkpoint.pth

        # Mixed: video input, frame output
        python demo/restoration_video_demo.py input.mp4 output_frames/ checkpoint.pth
    """
    args = parse_args()

    # Check if input is video or image directory
    input_path = Path(args.input)
    is_video_input = input_path.suffix.lower() in VIDEO_EXTENSIONS

    # Read input frames
    print(f"Reading input from: {args.input}")
    if is_video_input:
        frames, input_fps = read_video_frames(str(input_path))
        print(f"Read {len(frames)} frames from video (fps: {input_fps:.2f})")
    else:
        frames = read_image_frames(str(input_path))
        input_fps = args.fps
        print(f"Read {len(frames)} frames from directory")

    if not frames:
        raise ValueError("No frames read from input")

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    if args.config:
        config = args.config
    else:
        config = get_default_gan_inference_config()
        print("Using default GAN config")

    device = torch.device(args.device)
    model = load_model(config, args.checkpoint, device)
    print(f"Model loaded on {device}")

    # Run inference
    print(f"Running inference on {len(frames)} frames...")
    output_frames = inference(model, frames, device, max_frames=args.max_frames)
    print(f"Inference complete. Generated {len(output_frames)} frames")

    # Write output
    output_path = Path(args.output)
    is_video_output = output_path.suffix.lower() in VIDEO_EXTENSIONS

    print(f"Writing output to: {args.output}")
    if is_video_output:
        # Use input fps if available, otherwise use specified fps
        output_fps = input_fps if is_video_input else args.fps
        write_video_frames(output_frames, str(output_path), fps=output_fps)
    else:
        write_image_frames(output_frames, str(output_path))

    print("Done!")


if __name__ == '__main__':
    main()
