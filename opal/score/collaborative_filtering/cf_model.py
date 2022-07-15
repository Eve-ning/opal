from typing import List

import pandas as pd
from surprise import Reader, Dataset, KNNWithMeans, Prediction
from surprise.dataset import DatasetAutoFolds
from surprise.model_selection import GridSearchCV
from tqdm import tqdm

from opal.score.collaborative_filtering.conf import PRED_VARIABLE


class CFModel:
    ds: DatasetAutoFolds

    def __init__(self, df: pd.DataFrame):
        """ Creates a Collaborative Filtering Model

        Args:
            df: Dataframe to be used to fit
        """
        self.reader = Reader(
            rating_scale=(
                float(df[PRED_VARIABLE].min()),
                float(df[PRED_VARIABLE].max())
            )
        )
        self.ds = Dataset.load_from_df(df, self.reader)
        self.algo = None

    def fit(self,
            name: str = "cosine",
            min_support: int = 50,
            user_based: bool = False,
            k: int = 20):
        """ Fits the algorithm """

        sim_options = {
            "name": name,
            "min_support": min_support,
            "user_based": user_based,
        }
        self.algo = KNNWithMeans(sim_options=sim_options, k=k)
        self.algo.fit(self.ds.build_full_trainset())

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Predicts the NA column in the DataFrame """

        for ix, row in tqdm(df.iterrows(),
                            desc="Predicting: ",
                            total=len(df)):
            pred: Prediction = self.algo.predict(uid=row['user'],
                                                 iid=row['map'])
            df.loc[ix, PRED_VARIABLE] = pred.est
        return df

    def search_and_apply(self,
                         names: List[str],
                         min_supports: List[int],
                         user_baseds: List[bool],
                         ks: List[int],
                         algo_class=KNNWithMeans,
                         folds: int = 4) -> None:
        """ Uses Grid Search to find best params & applies it.

        Args:
            names: List of https://surprise.readthedocs.io/en/stable/similarities.html
            min_supports: List of integers
            user_baseds: True and or False
            ks: List of Ks
            algo_class: Any KNN https://surprise.readthedocs.io/en/stable/prediction_algorithms_package.html
            folds: Number of KFolds

        """
        sim_options = {"name": names,
                       "min_support": min_supports,
                       "user_based": user_baseds}

        param_grid = {"sim_options": sim_options, 'k': ks}

        gs = GridSearchCV(algo_class, param_grid, cv=folds)
        gs.fit(self.ds)

        print(f"Mean Squared Error: ", gs.best_score["rmse"])
        print(f"Best Parameters: ", gs.best_params["rmse"])
        print(f"Applying Parameters")

        params = gs.best_params["rmse"]
        self.fit(params['sim_options']['name'],
                 params['sim_options']['min_support'],
                 params['sim_options']['user_based'],
                 params['k'])
