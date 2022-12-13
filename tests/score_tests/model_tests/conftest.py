import sys

import pytest

from opal.conf.conf import ROOT_DIR


@pytest.fixture(scope='session', autouse=True)
def add_submod_path():
    sys.path.append((ROOT_DIR / "data_ppy_sh_to_csv").as_posix())
