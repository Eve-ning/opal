from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
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


def similarity_pair(df: pd.DataFrame,
                    min_pair_plays: int = 40) -> pd.DataFrame:
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

    return df_sim
