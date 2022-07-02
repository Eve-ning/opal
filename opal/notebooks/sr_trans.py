import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("0.15,0.2,(2019, 2022),4.5,(600000, 990000),True,None.csv",
                 index_col=0)
# %%
val = (df['value_x'].values - 600000) / (990000 - 600000)
val = (1 / val)
val += df['difficultyrating'].mean() - val.mean()

#%%
df['sr_new'] = val

df_out = pd.DataFrame()
df_out['sr_new'] = df['sr_new']
df_out['sr_old'] = df['difficultyrating']
df_out['delta'] = df_out['sr_new'] - df_out['sr_old']
df_out['filename'] = df['filename']
