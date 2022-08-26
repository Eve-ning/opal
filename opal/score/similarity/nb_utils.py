""" Similarity Utils Class

This file is to hold separate functions to support the Python notebooks.

"""

from collections import namedtuple
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
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


SimilarityPairResult = namedtuple(
    'SimilarityPairResult',
    ['df', 'df_sim', 'df_support', 'qt']
)


def similarity_pair(
    df: pd.DataFrame,
) -> SimilarityPairResult:
    """ Finds the similarity pair within the score df.

    Notes:
        This uses Accuracy to find similarities between pairs

    Args:
        df: DataFrame that includes, accuracy, year, user_id & beatmap_id

    Returns:
        Similarity, Support, Score DFs and the fitted QuantileTransformer
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
    df_sim = pd.DataFrame(columns=ix, index=ix, data=ar_sim)
    df_sim.index.set_names(['user_id', 'year'])
    df_support = df_sim.copy(deep=True).astype(int)
    df_sim[:] = np.nan

    gb_pair = combinations(gb, 2)
    pair_n = int(ix_n * (ix_n - 1) / 2)

    for (pxi, df_px), (pyi, df_py) in tqdm(gb_pair, total=pair_n):
        # Find common maps played
        df_p = df_px.merge(df_py, on='map_id')
        support = len(df_p)
        df_support.loc[pxi, pyi] = support
        if support < 2:
            continue

        acc_x, acc_y = df_p['accuracy_qt_x'], df_p['accuracy_qt_y']

        # Calculate Sigma & Similarity Score
        sigma = fit_sigma(acc_x, acc_y)
        df_sim.loc[pxi, pyi] = sigma_to_sim(sigma)

    # Reflect on diagonal
    df_sim[df_sim.isna()] = df_sim.T
    df_support += df_support.T

    return SimilarityPairResult(df, df_sim, df_support, qt)


def similarity_predict(df: pd.DataFrame,
                       df_sim: pd.DataFrame,
                       df_support: pd.DataFrame,
                       qt: QuantileTransformer,
                       sim_weight_pow: float = 8,
                       min_sim_support: int = 2,
                       min_pair_support: int = 50,
                       post_correct_pow: float = 1.14) -> pd.DataFrame:
    dfs_user = []
    for ix, sim in tqdm(df_sim.iterrows(), total=len(df_sim)):
        # If there are less than min_support similarities, skip
        if (df_support[ix] < min_pair_support).sum() < min_sim_support:
            continue
        df_user = df.loc[ix].set_index('map_id')
        # Adds similarity as a column to the df
        #                        vvvv
        # +--------+--------+------------+
        # | map_id | acc_qt | similarity |
        # +--------+--------+------------+
        df_sim_user = pd.merge(
            df,
            sim[sim > 0.001].dropna().rename('similarity'),
            left_index=True, right_index=True
        )
        # Within each map, we find weighted average (weighted by similarity)
        # +--------+--------+------------+
        # | map_id | acc_qt | similarity |
        # +--------+---^----+------^-----+
        #     (target) |           | (weights)
        #              +-----------+
        if df_sim_user.empty:
            continue
        df_sim_user_g = df_sim_user.groupby('map_id')
        df_pred = df_sim_user_g.apply(
            lambda g: np.average(
                g['accuracy_qt'],
                weights=g['similarity'] ** sim_weight_pow
            )
        )
        # This will yield us a SINGLE prediction per map_id

        # We also COUNT the number of supports
        df_support_user = df_sim_user_g.agg('count').iloc[:, 0]

        # Join the prediction & support to the user df
        df_user = df_user.merge(df_pred.rename('predict_qt'),
                                left_index=True, right_index=True)
        df_user = df_user.merge(df_support_user.rename('support'),
                                left_index=True, right_index=True)

        # Inverse transform the prediction_qt
        df_user['predict'] = qt.inverse_transform(
            df_user[['predict_qt']].to_numpy()
        )

        dfs_user.append(df_user)

    return pd.concat(dfs_user) ** post_correct_pow


Evaluation = namedtuple('Evaluation', ['mse', 'r2'])


def evaluate(pred, actual) -> Evaluation:
    return Evaluation(mean_squared_error(actual, pred, squared=False),
                      r2_score(actual, pred))
