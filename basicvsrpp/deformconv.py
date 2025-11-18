# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import math
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init as init
from torch.nn.modules.utils import _pair, _single

class ModulatedDeformConv2d(nn.Module):
    """Modulated Deformable Convolution 2D.

    This is a base class for modulated deformable convolution. Subclasses should
    override the forward method to implement specific deformable convolution logic.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution. Default: 1.
        padding: Zero-padding added to both sides of the input. Default: 0.
        dilation: Spacing between kernel elements. Default: 1.
        groups: Number of blocked connections from input to output channels. Default: 1.
        deform_groups: Number of deformable groups. Default: 1.
        bias: If True, adds a learnable bias to the output. Default: True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        deform_groups: int = 1,
        bias: bool = True
    ) -> None:
        super(ModulatedDeformConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deform_groups = deform_groups
        self.with_bias = bias
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize the weights of the convolution layer."""
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x: Tensor, offset: Tensor, mask: Tensor) -> Tensor:
        """Forward pass for modulated deformable convolution.

        This base implementation raises NotImplementedError. Subclasses must
        override this method to provide specific deformable convolution logic.

        Args:
            x: Input tensor of shape (N, C, H, W).
            offset: Offset tensor for deformable convolution.
            mask: Modulation mask tensor.

        Returns:
            Output tensor after deformable convolution.

        Raises:
            NotImplementedError: This base class method must be overridden by subclasses.
        """
        raise NotImplementedError(
            "ModulatedDeformConv2d is a base class. "
            "Subclasses must implement the forward method."
        )