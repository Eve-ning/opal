import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

df = pd.read_csv("data/top_1000_map_scores.csv")
df_bms = pd.read_csv("data/beatmaps.csv", index_col='beatmap_id')
# %%
df_bm = df.groupby('beatmap_id')
# %%
fig, ax = plt.subplots(5, 2, figsize=(5, 10))

for bm_id, g in tqdm(df_bm):
    beatmap = df_bms.loc[bm_id]
    if beatmap.playmode != 3: continue
    if beatmap.approved != 1: continue
    g = g[g.enabled_mods == 0]
    od = min(int(beatmap.diff_overall), 9)
    ax_bm = ax[int(od) // 2][int(od) % 2]
    # ax_bm = ax
    if od == 6: color = 'black'
    if od == 7: color = 'red'
    if od == 8: color = 'green'
    if od == 9: color = 'blue'

    ax_bm.hist((g.score / 1e6),
               bins=100,
               histtype=u'step',
               # cumulative=1,
               # density=True,
               alpha=0.2,
               color='black')


fig.tight_layout()
fig.show()

# %%

for i in range(10):
    print([int(i) // 2],[int(i) % 2])

# %%
