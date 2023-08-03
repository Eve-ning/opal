import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
from urllib.parse import quote_plus

import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sqlalchemy import create_engine
from torch.utils.data import DataLoader, TensorDataset, random_split


@dataclass
class ScoreDataModule(pl.LightningDataModule):
    osu_files_path: Path
    train_test_val: Sequence[float] = field(default_factory=lambda: (0.8, 0.1, 0.1))
    transformer: QuantileTransformer = QuantileTransformer(output_distribution='normal')
    batch_size: int = 32
    visual_complexity_limit: float = 0.05
    metric: str = 'accuracy'

    uid_le: LabelEncoder = field(default_factory=LabelEncoder, init=False)
    mid_le: LabelEncoder = field(default_factory=LabelEncoder, init=False)

    def __post_init__(self):
        super().__init__()
        assert sum(self.train_test_val) == 1, "Train Test Validation must sum to 1."
        self.prepare_data()
        self.setup()

    def setup(self, stage: str = "") -> None:
        logging.info("Querying from DB")

        db_name: str = "osu"
        user_name: str = "root"
        password: str = "p@ssw0rd1"
        host: str = "opal.mysql"
        port: int = 3307
        quoted_password = quote_plus(password)
        engine = create_engine(f'mysql+mysqlconnector://{user_name}:{quoted_password}@{host}:{port}/{db_name}')

        con = engine.connect()
        df = pd.read_sql_table("opal_active_scores", con=con)
        df_vc = pd.read_sql_table("opal_beatmaps_visual_complexity", con=con)
        df_vc = df_vc.loc[df_vc['visual_complexity'] < self.visual_complexity_limit]

        df = df.merge(df_vc, on=['mid'])
        df = self.db.get_df_score().drop('visual_complexity', axis=1, errors='ignore')

        logging.info("Creating IDs")
        df = self.get_ids(df)

        logging.info("Scaling Metrics")
        df = self.scale_metric(df, self.transformer, self.metric)

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
    def scale_metric(df: pd.DataFrame,
                     transformer: TransformerMixin,
                     metric: str):
        """ Uses the scalers provided to scale the metric """
        return df.assign(
            **{metric: lambda x: transformer.fit_transform(x[[metric]].values)}
        )

    @staticmethod
    def get_ids(df: pd.DataFrame):
        """ Prepares the ids used in Collaborative Filtering Model

        Notes:
            Creates uid = {user_id}/{year}
                    mid = {beatmap_id}/{speed}
            Label Encodes uid & mid.
        """
        df = df.assign(
            uid=lambda x: x.uid.astype(str) + "/" + x.year.astype(str),
            mid=lambda x: x.mid.astype(str) + "/" + x.speed.astype(str)
        ).drop(['year', 'speed'], axis=1)
        return df

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
