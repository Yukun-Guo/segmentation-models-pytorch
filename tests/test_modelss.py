import pytest
import torch
import segmentation_models_pytorch as smp

encoder_names = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_32x8d',
    'resnext101_32x16d', 'resnext101_32x32d', 'resnext101_32x48d', 'dpn68',
    'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131', 'vgg11', 'vgg11_bn',
    'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'senet154',
    'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d',
    'se_resnext101_32x4d', 'densenet121', 'densenet169', 'densenet201',
    'densenet161', 'inceptionresnetv2', 'inceptionv4', 'efficientnet-b0',
    'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4',
    'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7', 'mobilenet_v2',
    'xception'
]
arch_names = [
    'Unet', 'UnetPlusPlus', 'MAnet', 'Linknet', 'FPN', 'PSPNet', 'DeepLabV3',
    'DeepLabV3Plus', 'PAN', 'TriNet'
]

smp.DeepLabV3


@pytest.mark.parametrize("encoder_name", encoder_names)
@pytest.mark.parametrize("arch_name", arch_names)
def test_model(encoder_name, arch_name):
    sample = torch.rand(2, 1, 384, 288)
    model = smp.__dict__[arch_name](
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=1,
        classes=12,
        activation=None,
        encoder_attention_type='scse',
    )
    y = model(sample, sample,
              sample) if arch_name == 'TriNet' else model(sample)
    assert y.shape == (2, 12, 384, 288)
