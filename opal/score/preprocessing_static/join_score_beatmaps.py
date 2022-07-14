from __future__ import annotations

import pandas as pd

from opal.score.dataset import Dataset
from opal.score.preprocessing_static.conf import RENAME_MAPPING


def join_score_beatmaps(ds: Dataset) -> None:
    """ Joins the filtered score and beatmap datasets together """

    df = pd.merge(ds.scores_filtered_df,
                  ds.beatmaps_filtered_df,
                  on='beatmap_id') \
        .rename(RENAME_MAPPING, axis=1)
    df.to_csv(ds.joined_filtered_csv_path, index=False)
