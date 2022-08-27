from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.preprocessing import QuantileTransformer
from tqdm import tqdm

SimilarityModelParams = namedtuple(
    'SimilarityModelParams',
    ['df_scores', 'df_sim', 'df_support', 'qt']
)

Evaluation = namedtuple('Evaluation', ['mse', 'r2'])


@dataclass
class SimilarityModel:
    similarity_weight_pow: float = 8
    post_correct_pow: float = 1.14
    params: SimilarityModelParams = SimilarityModelParams(
        None, None, None, None
    )
    min_pair_support: int = 10

    def fit(self, df: pd.DataFrame) -> SimilarityModel:
        self.fit_similarity_pair(df)
        return self

    def fit_similarity_pair(self, df: pd.DataFrame):
        """ Finds the similarity pair within the score df.

        Notes:
            This uses Accuracy to find similarities between pairs

        Args:
            df: DataFrame that includes, accuracy, year, user_id & beatmap_id
        """

        def sigma_to_sim(sigma: float):
            """ Converts Sigma to Similarity Score """
            return float(np.exp(-np.abs(sigma)))

        def fit_sigma(score_x: pd.Series, score_y: pd.Series) -> float:
            """ Fits the x^e^s function to x, y scores """

            def curve_fn(x, sigma):
                """ Curve to fit to Player X Y performances """
                return x ** np.exp(sigma)

            return curve_fit(curve_fn, score_x, score_y,
                             p0=(0,), bounds=((-6), (6,)))[0]

        # Yield necessary columns only
        df = df[['map_id', 'user_id', 'accuracy', 'year']]
        df = df.groupby(['user_id', 'year', 'map_id']).agg(
            'mean').reset_index()

        # Uniformize score
        qt = QuantileTransformer()
        # to_numpy: to avoid restricting to a column name
        df['accuracy_qt'] = qt.fit_transform(df[['accuracy']].to_numpy())

        # We group by the Year & User ID
        # A Group will thus contain a user's score for a year
        df = df.set_index(['user_id', 'year', 'map_id'])
        gb = df.groupby(['user_id', 'year'])

        ixs = [g[0] for g in gb]
        ix_n = len(ixs)

        # Prep the similarity array to be filled
        ix = pd.MultiIndex.from_tuples(ixs, names=["user_id", "year"])
        ar_sim = np.zeros([ix_n, ix_n], dtype=float)
        df_sim = pd.DataFrame(columns=ix, index=ix, data=ar_sim)
        df_sim.index.set_names(['user_id', 'year'])
        df_support = df_sim.copy(deep=True).astype(int)
        df_sim[:] = np.nan

        gb_pair = combinations(gb, 2)
        pair_n = int(ix_n * (ix_n - 1) / 2)

        for (pxi, df_px), (pyi, df_py) in tqdm(
            gb_pair, total=pair_n,
            desc="Fitting Similarity Pair"
        ):
            # Find common maps played
            df_p = df_px.merge(df_py, on='map_id')
            support = len(df_p)
            if support < self.min_pair_support:
                continue
            df_support.loc[pxi, pyi] = support
            acc_x, acc_y = df_p['accuracy_qt_x'], df_p['accuracy_qt_y']

            # Calculate Sigma & Similarity Score
            try:
                sigma = fit_sigma(acc_x, acc_y)
            except RuntimeError:  # Can't find fit
                continue
            df_sim.loc[pxi, pyi] = sigma_to_sim(sigma)

        # Reflect on diagonal
        df_sim[df_sim.isna()] = df_sim.T
        df_support += df_support.T

        # df_sim = df_sim[df_support >= self.min_similarity_support]
        self.params = SimilarityModelParams(df, df_sim, df_support, qt)

    def predict(self, user_id: int, year: int, map_id: int) -> float:
        """ Predict the accuracy of map for 1 user

        Args:
            user_id: User ID to predict
            year: Year to predict
            map_id: Map ID to predict

        Notes:
            Prediction will fail if the user fails to play enough
            common maps with other players.

        Returns:
            An Accuracy prediction for that map.
        """
        ix = (user_id, year)
        # We find the df of scores with similarities associated with
        # user_id & year
        df = pd.merge(
            # User Scores
            self.params.df_scores,
            # Similarities associated with ix
            self.params.df_sim.loc[ix].dropna().rename('similarity'),
        )

        # Get only map_id
        df = df[df['map_id'] == map_id]

        if df.empty:
            raise Exception("Not enough supports to infer score")

        # Evaluate & Predict the accuracy using Weighted Average.
        acc_qt_pred = np.average(
            df['accuracy_qt'],
            weights=df['similarity'] ** self.similarity_weight_pow
        )

        # Transform from QT space to real space.
        acc_pred = self.params.qt.inverse_transform([[acc_qt_pred]])[0, 0]
        return acc_pred ** self.post_correct_pow

    def predict_self(self) -> pd.DataFrame:
        """ Predict all scores it trained with.

        Notes:
            This is useful for benchmarking how well it has learned the train
            set. However, note that its accuracy is upper-bound as it used
            the train for test.

        Returns:
            A DataFrame appended with its own predictions.
        """
        dfs_user = []
        for ix, sim in tqdm(self.params.df_sim.iterrows(),
                            total=len(self.params.df_sim)):
            df_user = self.params.df_scores.loc[ix].set_index('map_id')
            # Adds similarity as a column to the df
            #                        vvvv
            # +--------+--------+------------+
            # | map_id | acc_qt | similarity |
            # +--------+--------+------------+
            df_sim_user = pd.merge(
                self.params.df_scores,
                sim.dropna().rename('similarity'),
                left_index=True, right_index=True
            )

            # If there are no similarities
            if df_sim_user.empty:
                continue

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
                    weights=g['similarity'] ** self.similarity_weight_pow
                )
            )
            # This will yield us a SINGLE prediction per map_id

            # We also COUNT the number of supports
            df_support = df_sim_user_g.agg('count').iloc[:, 0]

            # Join the prediction & support to the user df
            df_user = df_user.merge(df_pred.rename('predict_qt'),
                                    left_index=True, right_index=True)
            df_user = df_user.merge(df_support.rename('support'),
                                    left_index=True, right_index=True)

            # Inverse transform the prediction_qt
            df_user['predict'] = self.params.qt.inverse_transform(
                df_user[['predict_qt']].to_numpy()
            )

            dfs_user.append(df_user)

        return pd.concat(dfs_user)
