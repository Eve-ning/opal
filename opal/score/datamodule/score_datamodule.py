import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset, random_split

from opal.conf.conf import DATA_DIR
from opal.conf.mods import OsuMod
from osu_data_csv.main import get_dataset


@dataclass
class ScoreDataModule(pl.LightningDataModule):
    ds_yyyy_mm: str = "2022_10"
    ds_mode: str = "mania"
    ds_set: str = "1000"

    train_test_val: Sequence[float] = field(default_factory=lambda: (0.8, 0.1, 0.1))

    qt_accuracy: QuantileTransformer = QuantileTransformer(output_distribution="normal")

    batch_size: int = 32
    m_min_support: int = 50
    u_min_support: int = 50
    score_bounds: Tuple[float, float] = (5e5, 1e6)
    accuracy_bounds: Tuple[float, float] = (0.5, 1)
    keys: Tuple[int] = (4, 7)
    sr_bounds: Tuple[float, float] = (2.5, 10.0)

    metric: str = 'accuracy'

    limit_scores_read: int = None

    uid_le: LabelEncoder = field(default_factory=LabelEncoder, init=False)
    mid_le: LabelEncoder = field(default_factory=LabelEncoder, init=False)

    def __post_init__(self):
        super().__init__()
        assert sum(self.train_test_val) == 1, "Train Test Validation must sum to 1."

    def prepare_data(self) -> None:
        """ Downloads data via data_ppy_sh_to_csv submodule """
        get_dataset(
            year_month=self.ds_yyyy_mm,
            mode=self.ds_mode,
            set=self.ds_set,
            dl_dir=DATA_DIR,
            bypass_confirm='Y',
            cleanup='N',
            ignore_path=(Path(__file__).parent / "ignore_mapping.yaml").as_posix()
        )

    def setup(self, stage: str = "") -> None:
        ds_str = f"{self.ds_yyyy_mm}_01_performance_{self.ds_mode}_top_{self.ds_set}"

        csv_dir = DATA_DIR / ds_str / "csv"
        csv_score = csv_dir / "osu_scores_mania_high.csv"
        csv_map = csv_dir / "osu_beatmaps.csv"

        logging.info("Preparing Metric DF")
        df_metric = pd.read_csv(csv_score, nrows=self.limit_scores_read)
        df_metric = self.prep_metric(df_metric, self.score_bounds, self.accuracy_bounds)

        logging.info("Preparing Map DF")
        df_map = pd.read_csv(csv_map)
        df_map = self.prep_map(df_map, self.keys, self.sr_bounds)

        logging.info("Merging")
        df = pd.merge(df_map, df_metric, how='inner', on='beatmap_id')

        logging.info("Creating IDs")
        df = self.get_ids(df)
        logging.info("Preparing merged DF")
        df = self.drop_infrequent_ids(df, self.m_min_support, self.u_min_support)
        logging.info("Scaling Metrics")
        df = self.scale_metric(df, self.qt_accuracy, self.metric)
        logging.info("Encoding Ids")
        df = self.encode_ids(df)
        self.df = df

        logging.info("Creating Tensors")
        x_uid = torch.Tensor(df[['uid_le']].values).to(torch.int)
        x_mid = torch.Tensor(df[['mid_le']].values).to(torch.int)
        y = torch.Tensor(df[[self.metric]].values)
        ds = TensorDataset(x_uid, x_mid, y)

        logging.info("Splitting to Train Test Validation")
        n_train = int(len(df) * self.train_test_val[0])
        n_test = int(len(df) * self.train_test_val[1])
        n_val = len(df) - (n_train + n_test)

        self.train_ds, self.test_ds, self.val_ds = random_split(ds, (n_train, n_test, n_val))

    @staticmethod
    def prep_metric(df: pd.DataFrame,
                    score_bounds: Tuple[float, float] = (5e5, 1e6),
                    accuracy_bounds: Tuple[float, float] = (0.5, 1)):
        """ Prepares the Score (Metric) DF

        Notes:
            Adjusts scores decreased by EASY, NO_FAIL & HALF_TIME
            Adds Speed column based on speed:
                - HALF_TIME: -1
                - No Mod: 0
                - DOUBLE_TIME: 1
            Adds Year column
        """

        half_score = (
                ((df['enabled_mods'] & OsuMod.EASY) > 0) |
                ((df['enabled_mods'] & OsuMod.NO_FAIL) > 0) |
                ((df['enabled_mods'] & OsuMod.HALF_TIME) > 0)
        )
        half_speed = (df['enabled_mods'] & OsuMod.HALF_TIME) > 0
        double_speed = (df['enabled_mods'] & OsuMod.DOUBLE_TIME) > 0

        df = df.assign(
            score=lambda x: x.score * np.where(half_score, 2, 1),
            speed=0,
            year=lambda x: x['date'].str[:4],
            accuracy=lambda x: (
                    (
                            x.count50 * (50 / 320) +
                            x.count100 * (100 / 320) +
                            x.countkatu * (200 / 320) +
                            x.count300 * (300 / 320) +
                            x.countgeki * (320 / 320)
                    ) / (
                            x.countmiss +
                            x.count50 +
                            x.count100 +
                            x.countkatu +
                            x.count300 +
                            x.countgeki
                    )
            )
        ).assign(
            speed=lambda x: np.where(half_speed, -1, x.speed)
        ).assign(
            speed=lambda x: np.where(double_speed, 1, x.speed)
        )[['user_id', 'beatmap_id', 'year', 'score', 'accuracy', 'speed']]
        return df[df['score'].between(*score_bounds) & df['accuracy'].between(*accuracy_bounds)]

    @staticmethod
    def scale_metric(df: pd.DataFrame,
                     scaler: TransformerMixin,
                     metric: str):
        """ Uses the scalers provided to scale the metric """
        return df.assign(
            **{metric: lambda x: scaler.fit_transform(x[[metric]].values)}
        )

    @staticmethod
    def prep_map(df: pd.DataFrame,
                 diff_sizes: Tuple[int] = (4, 7),
                 sr_bounds: Tuple[float, float] = (2.5, 10.0)):
        """ Prepares the Map DF

        Notes:
            Removes all non-mania maps
        """

        df = df.loc[
            (df['playmode'] == 3) &
            (df['diff_size'].isin(diff_sizes)) &
            (df['difficultyrating'].between(*sr_bounds)),
            ['difficultyrating', 'diff_overall', 'diff_size', 'beatmap_id', 'filename']
        ]
        return df

    @staticmethod
    def get_ids(df: pd.DataFrame):
        """ Prepares the ids used in Collaborative Filtering Model

        Notes:
            Creates uid = {user_id}/{year}
                    mid = {beatmap_id}/{speed}
            Label Encodes uid & mid.
        """
        df = df.assign(
            uid=lambda x: x.user_id.astype(str) + "/" + x.year,
            mid=lambda x: x.beatmap_id.astype(str) + "/" + x.speed.astype(str)
        )  # [['uid', 'mid', 'score', 'accuracy']]
        return df

    @staticmethod
    def drop_infrequent_ids(df: pd.DataFrame,
                            m_min_support: int,
                            u_min_support: int):
        """ Drops infrequent ids for maps and users. """
        m_freq = df['mid'].value_counts()
        u_freq = df['uid'].value_counts()
        return df.mask(
            df['mid'].isin((m_freq[m_freq < m_min_support]).index)
        ).mask(
            df['uid'].isin((u_freq[u_freq < u_min_support]).index)
        ).dropna()

    def encode_ids(self, df: pd.DataFrame):
        """ Label Encode the ids """
        return df.assign(
            uid_le=lambda x: self.uid_le.fit_transform(x.uid),
            mid_le=lambda x: self.mid_le.fit_transform(x.mid),
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, shuffle=False, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds, shuffle=False, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.val_ds, shuffle=False, batch_size=self.batch_size)

    @property
    def n_uid(self):
        return len(self.uid_le.classes_)

    @property
    def n_mid(self):
        return len(self.mid_le.classes_)
