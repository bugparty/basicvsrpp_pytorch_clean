# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

from typing import Any, Optional, Tuple

from torch import Tensor

from basicvsrpp.mmagic.registry import MODELS
from basicvsrpp.mmagic.real_basicvsr import RealBasicVSR
from basicvsrpp.models.basicvsr_plusplus_net import BasicVSRPlusPlusGanNet

MODELS.register_module(module=BasicVSRPlusPlusGanNet)


@MODELS.register_module()
class BasicVSRPlusPlusGan(RealBasicVSR):
    """RealBasicVSR model for real-world video super-resolution.

    Ref:
    Investigating Tradeoffs in Real-World Video Super-Resolution, arXiv

    Args:
        generator (dict): Config for the generator.
        discriminator (dict, optional): Config for the discriminator.
            Default: None.
        gan_loss (dict, optional): Config for the gan loss.
            Note that the loss weight in gan loss is only for the generator.
        pixel_loss (dict, optional): Config for the pixel loss. Default: None.
        perceptual_loss (dict, optional): Config for the perceptual loss.
            Default: None.
        train_cfg (dict): Config for training. Default: None.
            You may change the training of gan by setting:
            `disc_steps`: how many discriminator updates after one generate
            update;
            `disc_init_steps`: how many discriminator updates at the start of
            the training.
            These two keys are useful when training with WGAN.
        test_cfg (dict): Config for testing. Default: None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`. Default: None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`. Default: None.
    """

    def __init__(
        self,
        generator: dict,
        discriminator: Optional[dict] = None,
        gan_loss: Optional[dict] = None,
        pixel_loss: Optional[dict] = None,
        perceptual_loss: Optional[dict] = None,
        is_use_ema: bool = False,
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
        data_preprocessor: Optional[dict] = None
    ) -> None:
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            gan_loss=gan_loss,
            pixel_loss=pixel_loss,
            perceptual_loss=perceptual_loss,
            is_use_sharpened_gt_in_pixel=False,
            is_use_sharpened_gt_in_percep=False,
            is_use_sharpened_gt_in_gan=False,
            is_use_ema=is_use_ema,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

    def extract_gt_data(self, data_samples: Any) -> Tuple[Tensor, Tensor, Tensor]:
        """Extract ground truth data from data samples.

        Args:
            data_samples: Data samples containing ground truth images.

        Returns:
            Tuple of (gt_pixel, gt_percep, gt_gan) tensors for different loss computations.
        """
        gt = data_samples.gt_img
        gt_pixel, gt_percep, gt_gan = gt.clone(), gt.clone(), gt.clone()
        n, t, c, h, w = gt_pixel.size()
        gt_pixel = gt_pixel.view(-1, c, h, w)
        gt_percep = gt_percep.view(-1, c, h, w)
        gt_gan = gt_gan.view(-1, c, h, w)

        return gt_pixel, gt_percep, gt_gan