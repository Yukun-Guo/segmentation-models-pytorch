import torch
import torch.nn as nn
from typing import List
from collections import OrderedDict

from . import _utils as utils


class EncoderMixin:
    """Add encoder functionality such as:
    - output channels specification of feature tensors (produced by encoder)
    - patching first convolution for arbitrary input channels
    """

    _output_stride = 32

    @property
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self._out_channels[:self._depth + 1]

    @property
    def output_stride(self):
        return min(self._output_stride, 2**self._depth)

    def set_in_channels(self, in_channels, pretrained=True):
        """Change first convolution channels"""
        if in_channels == 3:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] +
                                       list(self._out_channels)[1:])

        utils.patch_first_conv(model=self,
                               new_in_channels=in_channels,
                               pretrained=pretrained)

    def get_stages(self):
        """Override it in your implementation"""
        raise NotImplementedError

    def make_dilated(self, output_stride):
        if output_stride == 16:
            stage_list = [5]
            dilation_list = [2]
        elif output_stride == 8:
            stage_list = [4, 5]
            dilation_list = [2, 4]
        else:
            raise ValueError(
                f"Output stride should be 16 or 8, got {output_stride}.")
        self._output_stride = output_stride
        stages = self.get_stages()
        for stage_indx, dilation_rate in zip(stage_list, dilation_list):
            utils.replace_strides_with_dilation(module=stages[stage_indx],
                                                dilation_rate=dilation_rate)


class SCSEModule(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        tmp_out_chn = max(in_channels // reduction, 1)
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, tmp_out_chn, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(tmp_out_chn, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)
