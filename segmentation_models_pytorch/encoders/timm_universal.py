import timm
import torch.nn as nn
from ._base import SCSEModule


class TimmUniversalEncoder(nn.Module):

    def __init__(self,
                 name,
                 pretrained=True,
                 in_channels=3,
                 depth=5,
                 output_stride=32,
                 attention=None):
        super().__init__()
        kwargs = dict(
            in_chans=in_channels,
            features_only=True,
            output_stride=output_stride,
            pretrained=pretrained,
            out_indices=tuple(range(depth)),
        )

        # not all models support output stride argument, drop it by default
        if output_stride == 32:
            kwargs.pop("output_stride")

        self.model = timm.create_model(name, **kwargs)

        self._in_channels = in_channels
        self._out_channels = [
            in_channels,
        ] + self.model.feature_info.channels()
        self._depth = depth
        self._output_stride = output_stride

        self._attention = []  # 2/4
        if attention == 'scse':  # 3/4
            self._attention = nn.ModuleList([
                SCSEModule(chn) for chn in self.model.feature_info.channels()
            ])
            # skip first conv layer

    def forward(self, x):
        features_ = self.model(x)
        features = []
        for i, f in enumerate(features_):
            if self._attention:  # 4/4
                f = self._attention[i](f)
            features.append(f)

        features = [
            x,
        ] + features

        return features

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def output_stride(self):
        return min(self._output_stride, 2**self._depth)
