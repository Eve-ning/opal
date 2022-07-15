from pathlib import Path
from typing import Tuple

import pandas as pd
import pytest

THIS_DIR = Path(__file__).parent


def pickled_data(pkl_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_pickle(pkl_path).fillna(0)
    return df.drop('error', axis=1), df['error']


@pytest.fixture
def train_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return pickled_data(THIS_DIR / "train_test.pkl")


@pytest.fixture
def validation_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return pickled_data(THIS_DIR / "validation.pkl")
