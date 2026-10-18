# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
from torch import Tensor, nn

from basicvsrpp.checkpoint import load_state_dict
from basicvsrpp.models.basicvsr_plusplus_net import BasicVSRPlusPlusGanNet, BasicVSRPlusPlusNet
from lib.image_utils import img2tensor, tensor2img

_GENERATORS = {
    'BasicVSRPlusPlusNet': BasicVSRPlusPlusNet,
    'BasicVSRPlusPlusGanNet': BasicVSRPlusPlusGanNet,
}
_MODEL_TYPES = {'BasicVSR', 'RealBasicVSR', 'BasicVSRPlusPlusGan'}


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


class VideoRestorationModel(nn.Module):
    """Inference-only model: holds just the generator whose weights are used at test time.

    The generator is registered under its checkpoint name (``generator`` or
    ``generator_ema``), so state dict keys match the training checkpoint.
    """

    def __init__(self, generator: nn.Module, weights_key: str) -> None:
        super().__init__()
        self.weights_key = weights_key
        self.add_module(weights_key, generator)

    def forward(self, inputs: Tensor) -> Tensor:
        return getattr(self, self.weights_key)(inputs)


def build_inference_model(config: Dict[str, Any]) -> VideoRestorationModel:
    model_type = config.get('type')
    if model_type not in _MODEL_TYPES:
        raise ValueError(f"unsupported model type {model_type!r}, expected one of {sorted(_MODEL_TYPES)}")
    generator_cfg = dict(config['generator'])
    generator_type = generator_cfg.pop('type')
    if generator_type not in _GENERATORS:
        raise ValueError(f"unsupported generator type {generator_type!r}, expected one of {sorted(_GENERATORS)}")
    # Models trained with EMA run inference with the EMA copy of the generator.
    weights_key = 'generator_ema' if config.get('is_use_ema', False) else 'generator'
    return VideoRestorationModel(_GENERATORS[generator_type](**generator_cfg), weights_key)


def load_model(
    config: Union[str, dict],
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device]
) -> VideoRestorationModel:
    """Load a model from config and checkpoint.

    Args:
        config: A dict definition of the model, or the path of an mmengine config
            file (reading config files requires mmengine).
        checkpoint_path: Path or URL of the checkpoint.
        device: Device to load the model on (e.g., 'cpu', 'cuda', or torch.device).

    Returns:
        The loaded model in evaluation mode.

    Raises:
        TypeError: If config is not a str or dict.
        KeyError: If the checkpoint has no weights for the generator used at inference.
        RuntimeError: If the generator weights do not match the network.
    """
    if isinstance(device, str):
        device = torch.device(device)

    if isinstance(config, str):
        from mmengine.config import Config
        config = Config.fromfile(config).model
    elif not isinstance(config, dict):
        raise TypeError(
            f"unsupported type for 'config': {type(config).__name__}. "
            "Must be either a file path (str) to a config file or a dict definition of the model"
        )

    model = build_inference_model(config)
    prefix = f'{model.weights_key}.'
    weights = {k: v for k, v in load_state_dict(checkpoint_path).items() if k.startswith(prefix)}
    if not weights:
        raise KeyError(f"checkpoint {checkpoint_path} has no '{prefix}*' weights")
    model.load_state_dict(weights, strict=True)
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
        List of restored frames as numpy arrays with the same shape as the input.

    Raises:
        AssertionError: If output frame count or shape doesn't match input.
    """
    input_frame_count = len(video)
    input_frame_shape = video[0].shape

    if isinstance(device, str):
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

        assert input_frame_count == output_frame_count and input_frame_shape == output_frame_shape, \
            f"expected {input_frame_count} frames of {input_frame_shape}, " \
            f"got {output_frame_count} frames of {output_frame_shape}"

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
