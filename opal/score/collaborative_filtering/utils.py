from typing import Tuple

import numpy as np
import pandas as pd

from opal.score.collaborative_filtering.conf import PRED_VARIABLE


def train_na_pivot(df: pd.DataFrame) \
    -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Creates the train test and NA dataframes.

    Notes:
        The train comes from non NA scores.
        NA are scores that aren't set.

    """

    df_pivot = df.pivot_table(
        index=['user_id', 'year', 'mod'],
        columns='map_id',
        values=PRED_VARIABLE,
        aggfunc=np.max
    )

    # We unpivot to yield the multi index
    df_unpivot = df_pivot.reset_index().melt(
        id_vars=['user_id', 'year', 'mod'],
        value_name=PRED_VARIABLE
    )

    # We generate a user from the multiindex (id + mod)
    df_unpivot['user'] = (df_unpivot['user_id'].astype(str)) + "/" + \
                         (df_unpivot['year'].astype(str))
    df_unpivot['map'] = (df_unpivot['map_id'].astype(str)) + "/" + \
                        (df_unpivot['mod'].astype(str))

    # Create the surprise acceptable df
    df_surprise = df_unpivot[['user', 'map', PRED_VARIABLE]]
    df_na = df_surprise[df_surprise.isna().any(axis=1)]
    df_train = df_surprise.dropna()

    return df_train, df_na
