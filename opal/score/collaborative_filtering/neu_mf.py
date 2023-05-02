from __future__ import annotations

from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ExponentialLR

from opal.score.collaborative_filtering.neu_mf_module import NeuMFModule


class NeuMF(pl.LightningModule):
    def __init__(
            self,
            uid_le: LabelEncoder,
            mid_le: LabelEncoder,
            qt: QuantileTransformer,
            emb_dim: int,
            mf_repeats: int,
            mlp_range: List[int],
            lr: float = 0.005,
            lr_gamma: float = 0.25
    ):
        """ Initializes the model for training

        Notes:
            If you want to load the model, use `NeuMF.load_from_checkpoint("path/to/model.ckpt")`

        Args:
            uid_le: UID LabelEncoder from the DM
            mid_le: MID LabelEncoder from the DM
            qt: QuantileTransformer from the DM
            emb_dim: Embedding Dimensions
            lr: Learning Rate

        """
        super().__init__()
        self.model = NeuMFModule(
            n_uid=len(uid_le.classes_),
            n_mid=len(mid_le.classes_),
            mf_repeats=mf_repeats,
            emb_dim=emb_dim,
            mlp_range=mlp_range
        )
        self.loss = MSELoss()
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.uid_le = uid_le
        self.mid_le = mid_le
        self.qt = qt

        # Save the params in the hparams.yaml
        self.save_hyperparameters()

    def forward(self, uid, mid):
        return self.model(uid, mid)

    def scaler_inverse_transform(self, val: torch.Tensor):
        return self.qt.inverse_transform(val.detach().cpu().numpy())

    def uid_inverse_transform(self, val: torch.Tensor):
        return self.uid_le.inverse_transform(val.detach().cpu().numpy())

    def mid_inverse_transform(self, val: torch.Tensor):
        return self.mid_le.inverse_transform(val.detach().cpu().numpy())

    def training_step(self, batch, batch_idx):
        *_, y_pred, y_true, y_pred_real, y_true_real = self.step(batch)
        loss = self.loss(y_pred, y_true)

        # if batch_idx % 32 == 0:
        #     self.logger.experiment.add_histogram("pred", y_pred)
        #     self.logger.experiment.add_histogram("true", y_true)
        #     self.logger.experiment.add_histogram("pred_real", y_pred_real)
        #     self.logger.experiment.add_histogram("true_real", y_true_real)

        self.log("train_loss", loss)
        self.log("train_mae", np.abs(y_pred_real - y_true_real).mean(), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        *_, y_pred, y_true, y_pred_real, y_true_real = self.step(batch)
        loss = self.loss(y_pred, y_true)

        self.log("val_loss", loss)
        self.log("val_mae", np.abs(y_pred_real - y_true_real).mean(), prog_bar=True)

    def predict_step(self, batch, batch_idx, **kwargs):
        x_uid, x_mid, *_, y_pred_real, y_true_real = self.step(batch)

        x_uid_real = self.uid_inverse_transform(x_uid)
        x_mid_real = self.mid_inverse_transform(x_mid)

        return x_uid_real, x_mid_real, y_pred_real, y_true_real

    def predict(self, x_uid_real: str | List[str], x_mid_real: str | List[str]) -> np.ndarray:
        """ Predicts the accuracy for a uid and mid

        Args:
            x_uid_real: The UID of the player in the format {PLAYER_ID}/{YEAR}
            x_mid_real: The MID of the player in the format {MAP_ID}/{SPEED}. SPEED can be -1, 0 or 1.
                -1 half-time, 0 normal time, 1 double time.

        Examples:
            Given that we want to predict user 12345, for year 2020, on the map 54321, with double time.
            >>> model.predict("12345/2020", "54321/2")

        Raises:
            ValueError if the model cannot predict the score.
        """
        x_uid_real = [x_uid_real] if isinstance(x_uid_real, str) else x_uid_real
        x_mid_real = [x_mid_real] if isinstance(x_mid_real, str) else x_mid_real

        x_uid = torch.Tensor(self.uid_le.transform(x_uid_real)[np.newaxis, :]).to(int).T.to(
            'cuda' if torch.cuda.is_available() else 'cpu')
        x_mid = torch.Tensor(self.mid_le.transform(x_mid_real)[np.newaxis, :]).to(int).T.to(
            'cuda' if torch.cuda.is_available() else 'cpu')

        return self.scaler_inverse_transform(self(x_uid, x_mid)).squeeze()

    def step(self, batch):
        x_uid, x_mid, y_true = batch
        y_pred = self(x_uid, x_mid)
        y_pred_real = self.scaler_inverse_transform(y_pred)
        y_true_real = self.scaler_inverse_transform(y_true)

        return x_uid, x_mid, y_pred, y_true, y_pred_real, y_true_real

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.001)

        return [optim], [
            {
                "scheduler": ExponentialLR(optim, self.lr_gamma),
                "interval": "epoch",
                "frequency": 1
            },
        ]
