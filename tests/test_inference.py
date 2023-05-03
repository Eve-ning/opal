import pytest

from opal import OpalNet


@pytest.fixture(scope="session")
def opal():
    return OpalNet.load(is_eval=True)


def test_inference_single(opal):
    assert opal.predict("2193881/2020", "767046/0") > 0.9


def test_inference_list(opal):
    preds = opal.predict(["2193881/2020", "2193881/2017"], ["767046/0", "767046/1"])
    assert all([p > 0.9 for p in preds])


def test_missing_inference(opal):
    with pytest.raises(ValueError):
        opal.predict("2193881/2018", "767046/0")

