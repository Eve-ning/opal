import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

from opal.score.similarity import SimilarityModel


def similarity_model_eval(df_score,
                          k_folds: int = 5,
                          k_fold_seed: int = 0,
                          **sim_kwargs) -> pd.DataFrame:
    """ Evaluates the score df on the similarity model.

    Args:
        df_score: Score DF
        k_folds: Number of K Folds to perform
        k_fold_seed: Random State seed for KFold
        **sim_kwargs: Kwargs for SimilarityModel

    """
    kf = KFold(shuffle=True, random_state=k_fold_seed, n_splits=k_folds)
    actuals_l = []
    preds_l = []
    for train_ix, test_ix in kf.split(df_score):
        train_df, test_df = df_score.iloc[train_ix], df_score.iloc[test_ix],
        sim = SimilarityModel(**sim_kwargs).fit(train_df)

        actuals, preds = [], []
        for ix, test_score in tqdm(test_df.iterrows(),
                                   total=len(test_df),
                                   desc="Predicting Scores"):
            try:
                pred = sim.predict(
                    test_score['user_id'],
                    test_score['year'],
                    test_score['map_id']
                )
            except (ZeroDivisionError, Exception):
                continue
            actuals.append(test_score['accuracy'])
            preds.append(pred)

        actuals_l.append(actuals)
        preds_l.append(preds)

    dfs = []
    for e, (actuals, preds) in enumerate(zip(actuals_l, preds_l)):
        df = pd.DataFrame([actuals, preds]).T
        df.columns = ['actual', 'pred']
        df['k'] = e
        dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True)
