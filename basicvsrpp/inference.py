# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import logging
from typing import Any, Union, Optional, List
from pathlib import Path

import numpy as np
import torch
from torch import nn
from basicvsrpp.mmagic.registry import MODELS
from basicvsrpp import register_all_modules
from mmengine.config import Config
from mmengine.runner import load_checkpoint

from lib.image_utils import img2tensor, tensor2img

logger = logging.getLogger(__name__)

def get_default_gan_inference_config() -> dict:
    return dict(
        type='BasicVSRPlusPlusGan',
        generator=dict(
            type='BasicVSRPlusPlusGanNet',
            mid_channels=64,
            num_blocks=15,
            spynet_pretrained=None),
        pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
        is_use_ema=True,
        data_preprocessor=dict(
            type='DataPreprocessor',
            mean=[0., 0., 0.],
            std=[255., 255., 255.],
        ))


def load_model(
    config: Union[str, dict, None],
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device]
) -> nn.Module:
    """Load a model from config and checkpoint.

    Args:
        config: Either a file path to a config file, a dict definition of the model, or None.
        checkpoint_path: Path to the model checkpoint file.
        device: Device to load the model on (e.g., 'cpu', 'cuda', or torch.device).

    Returns:
        The loaded model in evaluation mode.

    Raises:
        TypeError: If config is not a str, dict, or None.
    """
    register_all_modules()

    if device and isinstance(device, str):
        device = torch.device(device)

    if isinstance(config, str):
        config = Config.fromfile(config).model
    elif isinstance(config, dict):
        # Config is already a dict, use it as-is
        pass
    else:
        raise TypeError(
            f"unsupported type for 'config': {type(config).__name__}. "
            "Must be either a file path (str) to a config file or a dict definition of the model"
        )

    model = MODELS.build(config)
    load_checkpoint(model, checkpoint_path, map_location='cpu', logger=logger)
    model.cfg = config
    model.to(device)
    model.eval()
    return model


def inference(
    model: nn.Module,
    video: List[np.ndarray],
    device: Union[str, torch.device],
    max_frames: int = -1
) -> List[np.ndarray]:
    """Run inference on a video sequence.

    Args:
        model: The BasicVSR++ model to use for inference.
        video: List of video frames as numpy arrays (H, W, C) in BGR format.
        device: Device to run inference on (e.g., 'cpu', 'cuda', or torch.device).
        max_frames: Maximum number of frames to process at once. If > 0, the video
            will be processed in batches. If -1, all frames are processed together.
            Default: -1.

    Returns:
        List of output frames as numpy arrays (H*4, W*4, C) in BGR format, where
        the resolution is 4x upsampled compared to input.

    Raises:
        AssertionError: If output frame count or shape doesn't match input.
    """
    input_frame_count = len(video)
    input_frame_shape = video[0].shape

    if device and isinstance(device, str):
        device = torch.device(device)

    with torch.no_grad():
        result = []
        input_tensor = torch.stack(img2tensor(video, bgr2rgb=False, float32=True), dim=0)
        input_tensor = torch.unsqueeze(input_tensor, dim=0)  # TCHW -> BTCHW

        if max_frames > 0:
            for i in range(0, input_tensor.shape[1], max_frames):
                output = model(inputs=input_tensor[:, i:i + max_frames].to(device))
                result.append(output)
            result = torch.cat(result, dim=1)
        else:
            result = model(inputs=input_tensor.to(device))

        result = torch.squeeze(result, dim=0)  # BTCHW -> TCHW
        result_list = list(torch.unbind(result, 0))
        output = tensor2img(result_list, rgb2bgr=False, out_type=np.uint8, min_max=(0, 1))

        output_frame_count = len(output)
        output_frame_shape = output[0].shape
        assert input_frame_count == output_frame_count and input_frame_shape == output_frame_shape

        return output


def test() -> None:
    """Test function for local development and debugging."""
    device = "cuda:0"

    model = load_model(
        "configs/basicvsrpp/mosaic_restoration_generic_stage2.py",
        "experiments/basicvsrpp/mosaic_restoration_generic_stage2/iter_100000.pth",
        device
    )

    frame1 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    frame2 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    frame3 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    frame4 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    video = [frame1, frame2, frame3, frame4]
    result = inference(model, video, device)
    print(len(result), result[0].shape)


if __name__ == '__main__':
    test()
