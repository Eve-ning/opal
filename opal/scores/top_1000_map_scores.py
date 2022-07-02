# %%

import numpy as np
import pandas as pd
from surprise import KNNWithMeans, Reader, Dataset, KNNWithZScore, Prediction
from surprise.model_selection import GridSearchCV

from opal.conf.mods import OsuMod

# %%
df = pd.read_csv("data/top_1000_map_scores.csv")
# %%
df = df[['user_id', 'beatmap_id', 'score', 'enabled_mods', 'date', 'pp']]
df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
df['year'] = df['date'].dt.year
# %%
df['mod'] = ""
df.loc[(df['enabled_mods'] &
        OsuMod.HALF_TIME > 0), 'mod'] = "HT"
df.loc[(df['enabled_mods'] &
        (OsuMod.DOUBLE_TIME | OsuMod.NIGHTCORE) > 0), 'mod'] = "DT"
# df['mod'] = (df['enabled_mods'] & OsuMod.HALF_TIME > 0) * HALF_TIME
# df['mod'] += (df['enabled_mods'] & \
#               (OsuMod.DOUBLE_TIME | OsuMod.NIGHTCORE) > 0) * DOUBLE_TIME
# %%
df['score_id'] = df['user_id'].astype(str) + " " + \
                 df['year'].astype(str)
df['beatmap_id'] = df['beatmap_id'].astype(str) + " " + \
                   df['mod'].astype(str)
df = df[['score_id', 'beatmap_id', 'score']]

#%%
df = df[(df.score > 700000) & (df.score < 960000)]
# %%

reader = Reader(rating_scale=(500000, 1000000))
ds = Dataset.load_from_df(df, reader)
# %%
sim_options = {
    "name": "msd",
    "min_support": 5,
    "user_based": False,
}
algo = KNNWithMeans(sim_options=sim_options, k=10)
algo.fit(ds.build_full_trainset())

# %%
algo.predict('2193881 2021', '646681 DT')

# %%
beatmap_ids = df['beatmap_id'].unique()
# %%

performance = []
for y in [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]:
    y_ = []
    for bid in beatmap_ids:
        pred: Prediction = algo.predict(f'2193881 {y}', bid)
        y_.append(None if pred.details['was_impossible'] else pred.est)
    performance.append(y_)

# %%
performance_df = pd.DataFrame(
    performance,
    columns=beatmap_ids).T.astype(int) // 1000
performance_df = performance_df[performance_df.columns[::-1]]
performance_df.columns = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
#%%
ar = np.asarray(performance)
ar_norm = np.linalg.norm(ar,axis=0)

#%%
df_out =pd.DataFrame(ar_norm, index=beatmap_ids)
#%%
df_out.to_csv("out.csv")

#%%

#%%
performance_df.loc["421066 "]
# %%

# %%
df_pivot_na: pd.DataFrame = df.pivot('player', 'map')
df_na = df_pivot_na.melt(ignore_index=False).reset_index().drop([None], axis=1)

for ix, row in df_na.iterrows():
    if np.isnan(row['value']):
        pred = algo.predict(row['player'], row['map'])
        df_na.iloc[ix, -1] = pred.est

df_pivot: pd.DataFrame = df_na.pivot('player', 'map')
df_pivot.to_clipboard()
# %%
df_out = df_pivot_na.copy(True) // 1000
df_out[df_pivot_na.isna()] = - df_pivot.to_numpy() // 1000
df_out.astype(int).droplevel(axis=1, level=0).to_clipboard()
# %%
df_pivot_na.isna()

# %%
# Finds best params

sim_options = {
    "name": ["msd",],
    "min_support": [5],
    "user_based": [False],
}

param_grid = {"sim_options": sim_options, 'k': [10]}

gs = GridSearchCV(KNNWithZScore, param_grid, cv=2)
gs.fit(ds)

print(gs.best_score["rmse"])
print(gs.best_params["rmse"])
