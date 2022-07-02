from dataclasses import dataclass

from opal.conf.conf import OSU_DATA_PATH
from opal.data import Dataset


@dataclass
class Beatmaps:
    data: Dataset


b = Beatmaps(Dataset(OSU_DATA_PATH))
#%%
df = b.data.beatmaps_df
#%%
df = b.data.scores_df

#%%