# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import torch
import torch.nn as nn
from torch.nn import init as init
from torch.nn.modules.utils import _pair, _single
import math

class ModulatedDeformConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deform_groups=1,
                 bias=True):
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

    def init_weights(self):
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

    def forward(self, x, offset, mask):
        """Forward function for ModulatedDeformConv2d.

        Args:
            x (Tensor): Input feature, shape (n, c_in, h, w).
            offset (Tensor): Offset for deformable convolution, shape
                (n, deform_groups*2*kernel_h*kernel_w, h, w).
            mask (Tensor): Mask for deformable convolution, shape
                (n, deform_groups*kernel_h*kernel_w, h, w).

        Returns:
            Tensor: Output feature, shape (n, c_out, h_out, w_out).
        """
        import torchvision
        return torchvision.ops.deform_conv2d(
            x, offset, self.weight, self.bias,
            self.stride, self.padding,
            self.dilation, mask)