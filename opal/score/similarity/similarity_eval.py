import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from tqdm import tqdm

from opal.score.similarity import SimilarityModel


def similarity_model_eval(df_score,
                          k_folds: int = 5,
                          k_fold_seed: int = 0,
                          **sim_kwargs) -> tuple[
    pd.DataFrame, SimilarityModel]:
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
        for _, test_score in tqdm(test_df.itertuples(),
                                  total=len(test_df),
                                  desc="Predicting Scores"):
            try:
                pred = sim.predict(
                    test_score.user_id,
                    test_score.year,
                    test_score.map_id
                )
            except (ZeroDivisionError, Exception):
                continue
            actuals.append(test_score.accuracy)
            preds.append(pred)

        actuals_l.append(actuals)
        preds_l.append(preds)

    dfs = []
    for e, (actuals, preds) in enumerate(zip(actuals_l, preds_l)):
        df = pd.DataFrame([actuals, preds]).T
        df.columns = ['actual', 'pred']
        df['k'] = e
        dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True), sim


def metric_eval(df_eval: pd.DataFrame):
    mse = mean_squared_error(df_eval['actual'], df_eval['pred'], squared=False)
    r2 = r2_score(df_eval['actual'], df_eval['pred'])
    return mse, r2


def plot_eval(df_eval: pd.DataFrame):
    # Create Scatter
    plt.subplot(211)
    sns.scatterplot(
        x='actual',
        y='pred',
        data=df_eval,
        s=1
    )
    _ = plt.plot([0, 1], [0, 1], color='black', linestyle='dotted',
                 label='Perfect Prediction')
    _ = plt.xlim([0.85, 1])
    _ = plt.ylim([0.85, 1])
    _ = plt.xlabel("Actual")
    _ = plt.ylabel("Predict")
    _ = plt.title("Prediction vs. Actual")
    _ = plt.legend()
    _ = plt.gca().xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
    _ = plt.gca().yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    # Create Error Dist Barplot
    plt.subplot(212)
    df_eval['error'] = df_eval['pred'] - df_eval['actual']
    df_eval['error2'] = df_eval['error'] ** 2
    dfg = df_eval.groupby(pd.cut(df_eval['actual'], bins=25)).mean()[
              ['error2']] ** 0.5
    sns.barplot(x=[f"{i.right:.0%}" for i in dfg.index],
                y=dfg['error2'],
                color='red', ci=None)

    _ = plt.xlabel("Accuracy")
    _ = plt.ylabel("MSE")
    _ = plt.title("MSE for each Bin of Accuracy")
    plt.gca().yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    plt.tight_layout()
    mse, r2 = metric_eval(df_eval)
    plt.gcf().text(0, 1, f"MSE: {mse:.2%} | R^2: {r2:.2f}")
