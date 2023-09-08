from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale

from opal import OpalNet

# %%

df_user = pd.read_csv("users.csv", delimiter="\t")
df_map = pd.read_csv("maps.csv", delimiter="\t")
# %%
ckpt_path = Path(
    "../models/V4/2023_08_01_performance_mania_top_10000_20230819163602.csv"
    "/lightning_logs/version_1/checkpoints/epoch=6-step=56056.ckpt"
)
net = OpalNet.load_from_checkpoint(ckpt_path)

# %%
# Get embedding as array
u_emb_wgt = net.model.u_emb.weight.detach().cpu().numpy()
m_emb_wgt = net.model.m_emb.weight.detach().cpu().numpy()
# %%
pca_components = 6
u_emb_pca = PCA(n_components=pca_components)
u_emb_ld = minmax_scale(u_emb_pca.fit_transform(u_emb_wgt))
explained_u_emb_var = u_emb_pca.explained_variance_ratio_.round(3)
print(f"Explained User Variance: {explained_u_emb_var}")

m_emb_pca = PCA(n_components=pca_components)
m_emb_ld = minmax_scale(m_emb_pca.fit_transform(m_emb_wgt))
explained_m_emb_var = m_emb_pca.explained_variance_ratio_.round(3)
print(f"Explained Map Variance: {explained_m_emb_var}")
#%%
plt.hist(u_emb_wgt[:, 1], bins=100)
plt.show()

# %%
df_uid = (
    pd.concat(
        [
            pd.DataFrame(
                [uid.split("/") for e, uid in enumerate(net.uid_le.classes_)],
                columns=["user_id", "year"]
            ).astype({'user_id': int}),
            pd.DataFrame(u_emb_ld).reset_index(drop=True).astype(float)
        ],
        # ignore_index=True,
        axis=1
    )
).merge(df_user, on="user_id")
#%%
df_mid = (
    pd.concat(
        [
            pd.DataFrame(
                [mid.split("/") for mid in net.mid_le.classes_],
                columns=["beatmap_id", "speed"]
            ).astype({'beatmap_id': int}),
            pd.DataFrame(m_emb_ld).reset_index(drop=True).astype(float)
        ],
        # ignore_index=True,
        axis=1
    )
).merge(df_map[['beatmap_id', 'filename', 'difficultyrating']], on="beatmap_id")
# %%
pd.DataFrame(u_emb_ld).astype(float)
 