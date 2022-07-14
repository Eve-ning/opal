from dataclasses import dataclass

from opal.conf.conf import DATA_DIR
from opal.data import Dataset


@dataclass
class Beatmaps:
    data: Dataset


b = Beatmaps(Dataset(DATA_DIR))
#%%
df = b.data.beatmaps_df
#%%
df = b.data.scores_df

#%%