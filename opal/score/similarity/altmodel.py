import pandas as pd
from surprise import Dataset, AlgoBase
from surprise import Reader
from surprise.model_selection import train_test_split


class ExponentialFit(AlgoBase):
    def __init__(self, sim_options={}, bsl_options={}):

        AlgoBase.__init__(self, sim_options=sim_options,
                          bsl_options=bsl_options)

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        # Compute baselines and similarities
        self.sim = self.compute_similarities()

        return self

    def compute_similarities(self):
        """Build the similarity matrix.

        The way the similarity matrix is computed depends on the
        ``sim_options`` parameter passed at the creation of the algorithm (see
        :ref:`similarity_measures_configuration`).

        This method is only relevant for algorithms using a similarity measure,
        such as the :ref:`k-NN algorithms <pred_package_knn_inpired>`.

        Returns:
            The similarity matrix."""

        if self.sim_options["user_based"]:
            n_x, yr = self.trainset.n_users, self.trainset.ir
        else:
            n_x, yr = self.trainset.n_items, self.trainset.ur

        min_support = self.sim_options.get("min_support", 1)

        args = [n_x, yr, min_support]

        # Each yr is a dict[list[tuple]]
        # {
        #     uid0: [(bid0, score), (bid1, score), ...],
        #     uid1: [(bid0, score), (bid1, score), ...],
        #     ...
        # }

    def estimate(self, u, i):
        return 1


# User/Map |   0   |   1   |   2   |   3   |
# ---------+-------+-------+-------+-------+
#    0     |  900  |  700  | [550] |  850 |
#    1     | [950] |  800  |  600  |  950  |


# Creation of the dataframe. Column names are irrelevant.
ratings_dict = {'player_id': [0, 0, 0, 1, 1, 1, 1],
                'map_id': [0, 1, 3, 1, 2, 3, 2],
                'score': [900, 700, 850, 800, 600, 950, 0]}
df = pd.DataFrame(ratings_dict)

# A reader is still needed but only the rating_scale param is required.
reader = Reader(rating_scale=(0, 1000))
# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(df, reader)
train, test = train_test_split(data, 0.000001, random_state=0, shuffle=False)
# %%

from itertools import combinations
import pandas as pd

df_ir = {k: pd.DataFrame(v, columns=['uid', 'score']).set_index('uid')
         for k, v in train.ir.items()}
# %%

from scipy.optimize import curve_fit
import numpy as np


def sigma_to_sim(sigma: float):
    """ Converts Sigma to Similarity Score """
    return float(np.exp(-np.abs(sigma)))


def fit_sigma(score_x: pd.Series, score_y: pd.Series) -> float:
    """ Fits the x^e^s function to x, y scores """

    def curve_fn(x, sigma):
        """ Curve to fit to Player X Y performances """
        return x ** np.exp(sigma)

    return curve_fit(curve_fn, score_x, score_y,
                     p0=(0,),
                     # bounds=((-6), (6,))
                     )[0]


def fit_sim(score_x: pd.Series, score_y: pd.Series) -> float:
    return sigma_to_sim(fit_sigma(score_x, score_y))


for (uid_i, df_i), (uid_j, df_j) in combinations(df_ir.items(), 2):
    df_i: pd.DataFrame
    df_j: pd.DataFrame
    df_common = df_i.merge(df_j, on='uid')
    if len(df_common) < 2:
        continue
    df_common //= 1e3
    sim = fit_sim(df_common['score_x'], df_common['score_y'])
    print(sim)
    pass

# %%
import matplotlib.pyplot as plt

x, y = np.random.random([2, 100])
plt.plot()
