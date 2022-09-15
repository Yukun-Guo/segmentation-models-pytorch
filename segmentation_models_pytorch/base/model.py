import torch
from . import initialization as init
from typing import Optional


class SegmentationModel(torch.nn.Module):

    def __init__(self,
                 model_type: Optional[str] = None,
                 share_encoder: Optional[str] = False) -> None:
        self.model_type = model_type  # None or 'tri'
        self.share_encoder = share_encoder  # True or False
        super().__init__()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def check_input_shape(self, x):

        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride +
                     1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride +
                     1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, *x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x[0])
        features = self.encoder(x[0])

        if self.model_type == 'tri':
            if self.share_encoder:
                features_l = self.encoder(x[1])
                features_r = self.encoder(x[2])
            else:
                features_l = self.encoder_l(x[1])
                features_r = self.encoder_r(x[2])
            features = [features, features_l, features_r]

        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            if self.model_type == 'tri':
                labels = self.classification_head(features[0][-1])
            else:
                labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x
