# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Shared helpers for the golden regression set in tests/golden/.

Weights and auxiliary inputs are derived only from (key, shape) and fixed seeds
with numpy's legacy RandomState, whose streams are guaranteed stable, so they
do not depend on any model code or init function that a refactor may change.
"""

import os
import zlib

import numpy as np
import torch

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden')
DATA_PATH = os.path.join(GOLDEN_DIR, 'golden.npz')
MANIFEST_PATH = os.path.join(GOLDEN_DIR, 'manifest.json')

WEIGHT_SEED = 20251018
NUM_FRAMES, HEIGHT, WIDTH = 5, 256, 288  # 288/4 = 72 is not a multiple of 32: exercises SPyNet resize
ALIGN_SIZE = 16
CHUNK_FRAMES = 3


def deterministic_param(key: str, shape) -> torch.Tensor:
    rs = np.random.RandomState((WEIGHT_SEED + zlib.crc32(key.encode())) % 2**32)
    shape = tuple(shape)
    if len(shape) >= 2:
        fan_in = int(np.prod(shape[1:]))
        arr = rs.normal(0.0, 1.0 / np.sqrt(fan_in), size=shape)
    else:
        arr = rs.uniform(-0.05, 0.05, size=shape)
    return torch.from_numpy(arr.astype(np.float32))


def build_state_dict(manifest: dict, buffers: dict) -> dict:
    sd = {}
    for entry in manifest['state_dict']:
        key = entry['key']
        if entry['kind'] == 'param':
            sd[key] = deterministic_param(key, entry['shape'])
        else:
            sd[key] = torch.from_numpy(np.array(buffers[key]))
    return sd


def sub_state_dict(sd: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def synthetic_video(num_frames: int = NUM_FRAMES, height: int = HEIGHT, width: int = WIDTH) -> np.ndarray:
    """Smooth textured background with moving shapes and a mosaic patch, (T, H, W, 3) uint8 BGR."""
    rs = np.random.RandomState(7)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    frames = []
    for t in range(num_frames):
        dx, dy = 3.0 * t, 1.5 * t
        b = 0.5 + 0.3 * np.sin((xx - dx) / 17.0) * np.cos((yy - dy) / 23.0)
        g = 0.3 + 0.4 * (xx / width)
        r = 0.4 + 0.3 * np.sin((xx + yy - 2 * dx) / 31.0)
        img = np.stack([b, g, r], axis=-1)
        disk = (xx - (80 + 4 * t)) ** 2 + (yy - (90 + 2 * t)) ** 2 < 30 ** 2
        img[disk] = [0.9, 0.2, 0.1]
        y0, x0 = 150 - 3 * t, 170 + 2 * t
        img[y0:y0 + 40, x0:x0 + 60] = [0.1, 0.8, 0.3]
        img = img + rs.normal(0.0, 0.02, size=img.shape)
        img = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
        my, mx, block = 96, 112, 16
        patch = img[my:my + 64, mx:mx + 64]
        for by in range(0, 64, block):
            for bx in range(0, 64, block):
                cell = patch[by:by + block, bx:bx + block]
                cell[...] = cell.reshape(-1, 3).mean(axis=0).astype(np.uint8)
        frames.append(img)
    return np.stack(frames)


def video_to_tensor(video: np.ndarray) -> torch.Tensor:
    """(T, H, W, C) uint8 -> (T, C, H, W) float32 in [0, 1], same as lib.image_utils.img2tensor."""
    return torch.from_numpy(np.ascontiguousarray(video.transpose(0, 3, 1, 2))).float() / 255.


def net_input(video: np.ndarray) -> torch.Tensor:
    """Batch of two clips (2, 3, C, H, W): frames 0-2 and frames 4,3,2."""
    frames = video_to_tensor(video)
    return torch.stack([frames[0:3], frames[[4, 3, 2]]])


def align_inputs():
    rs = np.random.RandomState(11)
    s = ALIGN_SIZE

    def t(*shape, std=1.0):
        return torch.from_numpy(rs.normal(0.0, std, size=shape).astype(np.float32))

    x = t(1, 128, s, s)
    extra_feat = t(1, 192, s, s)
    flow_1 = t(1, 2, s, s, std=3.0)
    flow_2 = t(1, 2, s, s, std=3.0)
    return x, extra_feat, flow_1, flow_2


def summarize(arr: np.ndarray, spatial_axes: tuple, stride: int) -> dict:
    """Strided subsample + full-resolution center crop + global stats of a large array."""
    sub_index = [slice(None)] * arr.ndim
    crop_index = [slice(None)] * arr.ndim
    for ax in spatial_axes:
        sub_index[ax] = slice(1, None, stride)
        c = arr.shape[ax] // 2
        crop_index[ax] = slice(c - 16, c + 16)
    a64 = arr.astype(np.float64)
    return {
        'sub': np.ascontiguousarray(arr[tuple(sub_index)]),
        'crop': np.ascontiguousarray(arr[tuple(crop_index)]),
        'stats': np.array([a64.mean(), a64.std(), a64.min(), a64.max()]),
    }
