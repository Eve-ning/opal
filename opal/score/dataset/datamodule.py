import logging

import pandas as pd
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader

from data_ppy_sh_to_csv.main import get_dataset, default_sql_names
from opal.conf.conf import DATA_DIR
from opal.conf.mods import OsuMod


class ScoreDataset(pl.LightningDataModule):
    def __init__(self, train_test_val=(0.8, 0.1, 0.1)):
        super().__init__()
        assert sum(train_test_val) == 1, "Train Test Validation must sum to 1."

        csv_dir = DATA_DIR / "2022_12_01_performance_mania_top_1000" / "csv"

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

        self.df_map = df_map
        self.df_score = df_score
        self.df = df

        # x = torch.Tensor(df.loc[:, ['user_id', 'beatmap_id']].values)
        # y = torch.Tensor(df['score'].values)
        # ds = TensorDataset(x, y)
        #
        # n_train = int(len(df) * train_test_val[0])
        # n_test = int(len(df) * train_test_val[1])
        # n_val = len(df) - (n_train + n_test)
        #
        # self.train_ds, self.test_ds, self.val_ds = random_split(ds, (n_train, n_test, n_val))

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
        df.loc[(df['enabled_mods'] & OsuMod.EASY) > 0] *= 2
        df.loc[(df['enabled_mods'] & OsuMod.NO_FAIL) > 0] *= 2
        df.loc[(df['enabled_mods'] & OsuMod.HALF_TIME) > 0] *= 2

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
        df = df[df['playmode'] == 3]
        df = df[
            ['difficultyrating', 'diff_overall',
             'diff_size', 'version', 'beatmap_id', 'filename']
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
        df['uid'] = df['user_id'].astype(str) + "/" + df['year']
        df['mid'] = df['beatmap_id'].astype(str) + "/" + df['speed'].astype(str)
        df = df[['uid', 'mid', 'score']]
        self.le_uid = LabelEncoder()
        self.le_mid = LabelEncoder()
        self.ss_score = StandardScaler()
        df.uid = self.le_uid.fit_transform(df.uid)
        df.mid = self.le_mid.fit_transform(df.mid)
        df.score = self.ss_score.fit_transform(df[['score']])
        return df

    def prepare_data(self) -> None:
        """ Downloads data via data_ppy_sh_to_csv submodule """
        get_dataset("2022_10", "mania", "1000",
                    DATA_DIR, 'Y', default_sql_names[:4],
                    cleanup='N', zip_csv_files='N')

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_ds, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.val_ds, shuffle=False)
