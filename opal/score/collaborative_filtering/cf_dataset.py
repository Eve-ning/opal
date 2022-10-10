from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, random_split

from opal.conf.conf import SCORES_DIR
from opal.score.dataset import Dataset
from opal.score.preprocessing_dynamic import PreprocessingDynamic


@dataclass
class CFDataset:
    data_path: Path = SCORES_DIR
    acc_filter: Tuple = (0.8, 1)
    uid_support: int = 25
    mid_support: int = 25
    train_size: float = 0.7
    val_size: float = 0.2
    batch_size: int = 256
    train_workers: int = 4
    val_workers: int = 2
    test_workers: int = 2

    def __post_init__(self):
        df: pd.DataFrame = PreprocessingDynamic(
            Dataset(self.data_path, "top1k").joined_filtered_df,
            unpopular_maps_thres=None,
            unpopular_plays_thres=None,
            sr_min_thres=0,
            acc_filter=self.acc_filter,
            score_filter=None
        ).filter(calc_acc=True)

        df = df.rename({'accuracy': 'acc', 'map_id': 'mid'}, axis=1)
        df['uid'] = df['user_id'].astype(str) + "/" + df['year'].astype(str)
        df = df[['uid', 'mid', 'acc']].reset_index(drop=True)
        df = df.groupby(['uid', 'mid']).agg('mean').reset_index()

        df = df[df.groupby('mid').mid.transform('count') >= self.mid_support]
        df = df[df.groupby('uid').uid.transform('count') >= self.uid_support]

        self.uid_le = LabelEncoder()
        df['uid_le'] = self.uid_le.fit_transform(df['uid'])
        self.mid_le = LabelEncoder()
        df['mid_le'] = self.mid_le.fit_transform(df['mid'])

        x_uid = Tensor(df[['uid_le']].values).to(int)
        x_mid = Tensor(df[['mid_le']].values).to(int)
        y = Tensor(df[['acc']].values)
        ds = TensorDataset(x_uid, x_mid, y)

        train_size = int(len(ds) * self.train_size)
        val_size = int(len(ds) * self.val_size)
        test_size = len(ds) - train_size - val_size

        train_set, val_set, test_set = random_split(ds, [train_size, val_size, test_size])
        self.train_dl = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.train_workers)
        self.val_dl = DataLoader(val_set, batch_size=self.batch_size, num_workers=self.val_workers)
        self.test_dl = DataLoader(test_set, batch_size=self.batch_size, num_workers=self.test_workers)

    @property
    def uid_no(self):
        return len(self.uid_le.classes_)

    @property
    def mid_no(self):
        return len(self.mid_le.classes_)
