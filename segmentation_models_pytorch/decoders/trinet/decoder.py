import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md


class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type,
                                       in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type,
                                       in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class TriNetDecoder(nn.Module):

    def __init__(self,
                 encoder_channels,
                 decoder_channels,
                 n_blocks=5,
                 use_batchnorm=True,
                 attention_type=None,
                 center=False):
        super().__init__()
        if n_blocks != len(decoder_channels):
            raise ValueError(
                f"Model depth is {n_blocks}, but you provide `decoder_channels` for {len(decoder_channels)} blocks."
            )

        encoder_channels = encoder_channels[1:]
        encoder_channels = encoder_channels[::-1]
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels
        if center:
            self.center = CenterBlock(head_channels,
                                      head_channels,
                                      use_batchnorm=use_batchnorm)

        else:
            self.center = nn.Identity()
        kwargs = dict(use_batchnorm=use_batchnorm,
                      attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch,
                         **kwargs) for in_ch, skip_ch, out_ch in zip(
                             in_channels, skip_channels, out_channels)
        ]

        self.blocks = nn.ModuleList(blocks)
        self.attention_l = md.Attention(attention_type,
                                        in_channels=encoder_channels[0] * 2)

        self.attention_r = md.Attention(attention_type,
                                        in_channels=encoder_channels[-1] * 2)

        self.conv_1 = md.Conv2dReLU(
            in_channels=encoder_channels[0] * 5,
            out_channels=encoder_channels[0],
            kernel_size=1,
            padding=0,
        )
        self.conv_c = md.Conv2dReLU(
            in_channels=encoder_channels[0],
            out_channels=encoder_channels[0],
            kernel_size=3,
            padding=1,
        )

    def forward(self, *features):
        # remove first skip with same spatial resolution
        features_m = features[0][1:]
        # reverse channels to start from head of encoder
        features_m = features_m[::-1]

        head_m = features_m[0]
        skips_m = features_m[1:]

        features_l = features[1][-1]
        features_r = features[2][-1]

        x = self.center(head_m)
        x1 = self.attention_l(torch.cat([x, features_l], dim=1))
        x2 = self.attention_r(torch.cat([x, features_r], dim=1))
        x = self.conv_1(torch.cat([x1, x2, x], dim=1))
        x = self.conv_c(x)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips_m[i] if i < len(skips_m) else None
            x = decoder_block(x, skip)
        return x
