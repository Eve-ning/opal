from __future__ import annotations

from pathlib import Path

import pandas as pd
from surprise import KNNWithMeans

from opal.score.collaborative_filtering.cf_model import CFModel
from opal.score.collaborative_filtering.conf import SEARCH_KS, SEARCH_MIN_SUPPORTS, \
    SEARCH_NAMES, SEARCH_USER_BASEDS, PRED_VARIABLE
from opal.score.collaborative_filtering.utils import train_na_pivot


def predict(df: pd.DataFrame, save_path: Path):
    df_train, df_na = train_na_pivot(df)

    cf = CFModel(df_train)
    cf.search_and_apply(
        ks=SEARCH_KS,
        min_supports=SEARCH_MIN_SUPPORTS,
        names=SEARCH_NAMES,
        algo_class=KNNWithMeans,
        user_baseds=SEARCH_USER_BASEDS
    )
    cf.predict(df_na)

    df_result = (
        pd.concat([df_na, df_train])
            .groupby(["map"])
            .agg(
            pred_median=(PRED_VARIABLE, 'median'),
            pred_std=(PRED_VARIABLE, 'std')
        )
            .reset_index()
            .assign(
            map_id=lambda df_: df_['map'].str[:-2].astype(int)
        )
            .merge(df, on='map_id')
        [['map_id', 'pred_median', 'pred_std',
          'map_file_name', 'sr', 'keys']]
            .drop_duplicates()
            .sort_values('pred_median')
    )

    save_path.parent.mkdir(exist_ok=True)
    df_result.to_csv(save_path)
