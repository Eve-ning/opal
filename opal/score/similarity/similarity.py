from itertools import combinations
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.preprocessing import QuantileTransformer
from tqdm import tqdm


def sigma_to_sim(sigma):
    """ Converts Sigma to Similarity Score """
    return float(np.exp(-np.abs(sigma)))


def fit_sigma(score_x: pd.Series, score_y: pd.Series) -> float:
    """ Fits the x^e^s function to x, y scores """

    def curve_fn(x, sigma):
        """ Curve to fit to Player X Y performances """
        return x ** np.exp(sigma)

    return curve_fit(curve_fn, score_x, score_y)[0]


def similarity_pair(
    df: pd.DataFrame,
    min_pair_plays: int = 40
) -> Tuple[pd.DataFrame, pd.DataFrame, QuantileTransformer]:
    """ Finds the similarity pair within the score df.

    Notes:
        This uses Accuracy to find similarities between pairs

    Args:
        df: DataFrame that includes, accuracy, year, user_id & beatmap_id
        min_pair_plays: Minimum number of common players the pair must have.
            If less than min_common_plays, similarity will be NaN

    Returns:
        The Similarity DataFrame
    """

    # Yield necessary columns only
    df = df[['map_id', 'user_id', 'accuracy', 'year']]
    df = df.groupby(['user_id', 'year', 'map_id']).agg('mean').reset_index()

    # Uniformize score
    qt = QuantileTransformer()
    # to_numpy: to avoid restricting to a column name
    df['accuracy_qt'] = qt.fit_transform(df[['accuracy']].to_numpy())

    # We group by the Year & User ID
    # A Group will thus contain a user's score for a year
    df = df.set_index(['user_id', 'year'])
    gb = df.groupby(df.index)

    ixs = [g[0] for g in gb]
    ix_n = len(ixs)

    # Prep the similarity array to be filled
    ix = pd.MultiIndex.from_tuples(ixs, names=["user_id", "year"])
    ar_sim = np.empty([ix_n, ix_n], dtype=float)
    ar_sim[:] = np.nan
    df_sim = pd.DataFrame(columns=ix, index=ix, data=ar_sim)
    df_sim.index.set_names(['user_id', 'year'])

    gb_pair = combinations(gb, 2)
    pair_n = int(ix_n * (ix_n - 1) / 2)

    for (pxi, df_px), (pyi, df_py) in tqdm(gb_pair, total=pair_n):
        # Find common maps played
        df_p = df_px.merge(df_py, on='map_id')

        # If common maps < MIN_COMMON_PLAYS
        if len(df_p) < min_pair_plays:
            continue

        acc_x, acc_y = df_p['accuracy_qt_x'], df_p['accuracy_qt_y']

        # Calculate Sigma & Similarity Score
        sigma = fit_sigma(acc_x, acc_y)
        df_sim.loc[pxi, pyi] = sigma_to_sim(sigma)

    # Reflect on diagonal
    df_sim[df_sim.isna()] = df_sim.T

    df = df.set_index(['user_id', 'year'])

    return df_sim, df, qt


def similarity_predict(df: pd.DataFrame,
                       df_sim: pd.DataFrame,
                       qt: QuantileTransformer,
                       sim_weight_pow: float = 8) -> pd.DataFrame:
    dfs_user = []
    for ix, sim in tqdm(df_sim.iterrows(), total=len(df_sim)):
        df_user = df.loc[ix].set_index('map_id')
        # Adds similarity as a column to the df
        #                        vvvv
        # +--------+--------+------------+
        # | map_id | acc_qt | similarity |
        # +--------+--------+------------+
        df_sim_user = pd.merge(
            df,
            sim.dropna().rename('similarity'),
            left_index=True, right_index=True
        )

        # Within each map, we find weighted average (weighted by similarity)
        # +--------+--------+------------+
        # | map_id | acc_qt | similarity |
        # +--------+---^----+------^-----+
        #     (target) |           | (weights)
        #              +-----------+
        df_sim_user_g = df_sim_user.groupby('map_id')
        df_pred = df_sim_user_g.apply(
            lambda g: np.average(
                g['accuracy_qt'],
                weights=g['similarity'] ** sim_weight_pow
            )
        )
        # This will yield us a SINGLE prediction per map_id

        # We also COUNT the number of supports
        df_supp = df_sim_user_g.agg('count').iloc[:, 0]

        # Join the prediction & support to the user df
        df_user = df_user.merge(df_pred.rename('predict_qt'),
                                left_index=True, right_index=True)
        df_user = df_user.merge(df_supp.rename('support'),
                                left_index=True, right_index=True)

        # Inverse transform the prediction_qt
        df_user['predict'] = qt.inverse_transform(
            df_user[['predict_qt']].to_numpy()
        )

        dfs_user.append(df_user)

    return pd.concat(dfs_user)
