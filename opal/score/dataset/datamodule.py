import logging
from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split

from data_ppy_sh_to_csv.main import get_dataset, default_sql_names
from opal.conf.conf import DATA_DIR
from opal.conf.mods import OsuMod


@dataclass
class ScoreDataModule(pl.LightningDataModule):
    ds_yyyy_mm: str = "2022_10"
    ds_mode: str = "mania"
    ds_set: str = "1000"

    train_test_val: Sequence[float] = field(default_factory=lambda: (0.8, 0.1, 0.1))

    le_uid: LabelEncoder = LabelEncoder()
    le_mid: LabelEncoder = LabelEncoder()
    ss_score: StandardScaler = StandardScaler()

    def __post_init__(self):
        super().__init__()
        ds_str = f"{self.ds_yyyy_mm}_01_performance_{self.ds_mode}_top_{self.ds_set}"
        assert sum(self.train_test_val) == 1, "Train Test Validation must sum to 1."

        csv_dir = DATA_DIR / ds_str / "csv"
        csv_score = csv_dir / "osu_scores_mania_high.csv"
        csv_map = csv_dir / "osu_beatmaps.csv"

        logging.info("Preparing Score DF")
        df_score = pd.read_csv(csv_score)
        df_score = self.prep_score(df_score)

        logging.info("Preparing Map DF")
        df_map = pd.read_csv(csv_map)
        df_map = self.prep_map(df_map)

        logging.info("Merging")
        df = pd.merge(df_map, df_score, how='inner', on='beatmap_id')

        logging.info("Creating IDs")
        df = self.prep_ids(df)

        x = torch.Tensor(df.loc[:, ['uid', 'mid']].values).to(torch.int)
        y = torch.Tensor(df['score'].values)
        ds = TensorDataset(x, y)

        n_train = int(len(df) * self.train_test_val[0])
        n_test = int(len(df) * self.train_test_val[1])
        n_val = len(df) - (n_train + n_test)

        self.train_ds, self.test_ds, self.val_ds = random_split(ds, (n_train, n_test, n_val))

    def prep_score(self, df: pd.DataFrame):
        """ Prepares the Score DF

        Notes:
            Adjusts scores decreased by EASY, NO_FAIL & HALF_TIME
            Adds Speed column based on speed:
                - HALF_TIME: -1
                - No Mod: 0
                - DOUBLE_TIME: 1
            Adds Year column
        """

        df.loc[((df['enabled_mods'] & OsuMod.EASY) > 0) |
               ((df['enabled_mods'] & OsuMod.NO_FAIL) > 0) |
               ((df['enabled_mods'] & OsuMod.HALF_TIME) > 0)] *= 2

        df['speed'] = 0
        df.loc[(df['enabled_mods'] & OsuMod.HALF_TIME) > 0, 'speed'] = -1
        df.loc[(df['enabled_mods'] & OsuMod.DOUBLE_TIME) > 0, 'speed'] = 1

        df['year'] = df['date'].str[:4]
        df = df[['user_id', 'beatmap_id', 'year', 'score', 'speed']]
        return df

    def prep_map(self, df: pd.DataFrame):
        """ Prepares the Map DF

        Notes:
            Removes all non-mania maps
        """
        df = df.loc[
            df['playmode'] == 3,
            ['difficultyrating', 'diff_overall', 'diff_size',
             'version', 'beatmap_id', 'filename']
        ]
        return df

    def prep_ids(self, df: pd.DataFrame):
        """ Prepares the ids used in Collaborative Filtering Model

        Notes:
            Creates uid = {user_id}/{year}
                    mid = {beatmap_id}/{speed}
            Label Encodes uid & mid.
            Standard Scales score.
        """
        df = df.assign(
            uid=lambda x: x.user_id.astype(str) + "/" + x.year,
            mid=lambda x: x.beatmap_id.astype(str) + "/" + x.speed.astype(str)
        )[['uid', 'mid', 'score']].assign(
            uid=lambda x: self.le_uid.fit_transform(x.uid),
            mid=lambda x: self.le_mid.fit_transform(x.mid),
            score=lambda x: self.ss_score.fit_transform(x[['score']]),
        )
        return df

    def prepare_data(self) -> None:
        """ Downloads data via data_ppy_sh_to_csv submodule """
        get_dataset(
            year_month=self.ds_yyyy_mm,
            mode=self.ds_mode,
            set=self.ds_set,
            dl_dir=DATA_DIR,
            bypass_confirm='Y',
            sql_names=",".join(default_sql_names[:4]),
            cleanup='N',
            zip_csv_files='N'
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_ds, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.val_ds, shuffle=False)

    @property
    def n_uid(self):
        return len(self.le_uid.classes_)

    @property
    def n_mid(self):
        return len(self.le_mid.classes_)
