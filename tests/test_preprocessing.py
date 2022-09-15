import sys
import mock
import numpy as np

# mock detection module
sys.modules["torchvision._C"] = mock.Mock()
import segmentation_models_pytorch as smp  # noqa


def _test_preprocessing(inp, out, **params):
    preprocessed_output = smp.encoders.preprocess_input(inp, **params)
    assert np.allclose(preprocessed_output, out)


def test_mean():
    inp = np.ones((32, 32, 3))
    out = np.zeros((32, 32, 3))
    mean = (1, 1, 1)
    _test_preprocessing(inp, out, mean=mean)


def test_std():
    inp = np.ones((32, 32, 3)) * 255
    out = np.ones((32, 32, 3))
    std = (255, 255, 255)
    _test_preprocessing(inp, out, std=std)


def test_input_range():
    inp = np.ones((32, 32, 3))
    out = np.ones((32, 32, 3))
    _test_preprocessing(inp, out, input_range=(0, 1))
    _test_preprocessing(inp * 255, out, input_range=(0, 1))
    _test_preprocessing(inp * 255, out * 255, input_range=(0, 255))


def test_input_space():
    inp = np.stack([np.ones((32, 32)), np.zeros((32, 32))], axis=-1)
    out = np.stack([np.zeros((32, 32)), np.ones((32, 32))], axis=-1)
    _test_preprocessing(inp, out, input_space="BGR")


def test_preprocessing_params():
    params = _extracted_from_test_preprocessing_params_3("resnet18")
    params = _extracted_from_test_preprocessing_params_3("tu-resnet18")


# TODO Rename this here and in `test_preprocessing_params`
def _extracted_from_test_preprocessing_params_3(arg0):
    # check default encoder params
    result = smp.encoders.get_preprocessing_params(arg0)
    assert result["mean"] == [0.485, 0.456, 0.406]
    assert result["std"] == [0.229, 0.224, 0.225]
    assert result["input_range"] == [0, 1]
    assert result["input_space"] == "RGB"

    return result
