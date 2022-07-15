import tarfile
from typing import Tuple

import pandas as pd
import pytest

from opal.conf.conf import PUBLIC_OSU_REPLAYS_DIR, DATA_PUBLIC_DIR
from opal.replay.preprocess_replay_error import preprocess_replay_error


def unpack_tarfile(tar_path, extract_path):
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_path)


@pytest.fixture()
def rep_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    tar_path = DATA_PUBLIC_DIR / "osu.tar.gz"
    unpack_tarfile(tar_path, DATA_PUBLIC_DIR)
    map_dirs = [d for d in PUBLIC_OSU_REPLAYS_DIR.iterdir() if d.is_dir()]
    df = pd.concat([preprocess_replay_error(d) for d in map_dirs])

    y = df['error']
    X = df.drop('error', axis=1)
    return X, y
