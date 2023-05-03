import pytest
import torch

from opal.conf.conf import MODEL_DIR
from opal.score.collaborative_filtering import NeuMF


@pytest.fixture(scope="session")
def net():
    return NeuMF.load_from_checkpoint(MODEL_DIR / "V2_2023_04/checkpoints/epoch=8-step=55773.ckpt",
                                      map_location=torch.device('cpu'))


def test_inference_single(net):
    assert net.predict("2193881/2020", "767046/0") > 0.9


def test_inference_list(net):
    preds = net.predict(["2193881/2020", "2193881/2017"], ["767046/0", "767046/1"])
    assert all([p > 0.9 for p in preds])


def test_missing_inference(net):
    with pytest.raises(ValueError):
        net.predict("2193881/2018", "767046/0")
