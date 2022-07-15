from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


@dataclass
class Dataset:
    data_path: str | Path
    score_set: str = "scores_top1k"

    @property
    def files_path(self) -> Path:
        return self.data_path / "files"

    @property
    def nsv_maps_path(self) -> Path:
        return self.files_path / "nsv"

    @property
    def sv_maps_path(self) -> Path:
        return self.files_path / "sv"

    @property
    def scores_filtered_csv_path(self) -> Path:
        return self.data_path / f"{self.score_set}_filtered.csv"

    @property
    def joined_filtered_csv_path(self) -> Path:
        return self.data_path / f"{self.score_set}_joined_filtered.csv"

    @property
    def nsv_ids(self) -> List[int]:
        return [int(f.stem) for f in self.nsv_maps_path.glob("*")]

    @property
    def beatmaps_csv_path(self) -> Path:
        return self.data_path / "beatmaps.csv"

    @property
    def beatmaps_filtered_csv_path(self) -> Path:
        return self.data_path / "beatmaps_filtered.csv"

    @property
    def beatmaps_df(self) -> pd.DataFrame:
        return pd.read_csv(self.beatmaps_csv_path)

    @property
    def beatmaps_filtered_df(self) -> pd.DataFrame:
        return pd.read_csv(self.beatmaps_filtered_csv_path)

    @property
    def scores_csv_path(self) -> Path:
        return self.data_path / f"{self.score_set}.csv"

    @property
    def scores_df(self) -> pd.DataFrame:
        return pd.read_csv(self.scores_csv_path)

    @property
    def scores_filtered_df(self) -> pd.DataFrame:
        return pd.read_csv(self.scores_filtered_csv_path, index_col=None)

    @property
    def joined_filtered_df(self) -> pd.DataFrame:
        return pd.read_csv(self.joined_filtered_csv_path, index_col=None)
