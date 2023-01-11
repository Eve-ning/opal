from __future__ import annotations

from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn import MSELoss
from torch.optim.lr_scheduler import OneCycleLR

from opal.score.collaborative_filtering.neu_mf_module import NeuMFModule
from opal.score.datamodule import ScoreDataModule


class NeuMF(pl.LightningModule):
    def __init__(
            self,
            dm: ScoreDataModule,
            mf_emb_dim, mlp_emb_dim, mlp_chn_out,
            lr: float = 0.005,
            one_cycle_lr_params: dict = {}
    ):
        super().__init__()
        self.model = NeuMFModule(dm.n_uid, dm.n_mid, mf_emb_dim, mlp_emb_dim, mlp_chn_out)
        self.loss = MSELoss()
        self.lr = lr

        self.dm = dm
        self.one_cycle_lr_params = one_cycle_lr_params
        self.save_hyperparameters(ignore=['dm'])

    def forward(self, uid, mid):
        return self.model(uid, mid)

    def scaler_inverse_transform(self, val: torch.Tensor):
        return self.dm.scaler_accuracy.inverse_transform(val.detach().cpu().numpy())

    def uid_inverse_transform(self, val: torch.Tensor):
        return self.dm.uid_le.inverse_transform(val.detach().cpu().numpy())

    def mid_inverse_transform(self, val: torch.Tensor):
        return self.dm.mid_le.inverse_transform(val.detach().cpu().numpy())

    def training_step(self, batch, batch_idx):
        *_, y_pred, y_true, y_pred_real, y_true_real = self.step(batch)
        loss = self.loss(y_pred, y_true)

        # if batch_idx % 32 == 0:
        #     self.logger.experiment.add_histogram("pred", y_pred)
        #     self.logger.experiment.add_histogram("true", y_true)

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
        """ Predicts a uid and mid

        Args:
            x_uid_real:
            x_mid_real:

        Returns:

        """
        x_uid_real = [x_uid_real] if isinstance(x_uid_real, str) else x_uid_real
        x_mid_real = [x_mid_real] if isinstance(x_mid_real, str) else x_mid_real

        x_uid = torch.Tensor(self.dm.uid_le.transform(x_uid_real)[np.newaxis, :]).to(int).T
        x_mid = torch.Tensor(self.dm.mid_le.transform(x_mid_real)[np.newaxis, :]).to(int).T

        return self.scaler_inverse_transform(self(x_uid, x_mid)).squeeze()

    def step(self, batch):
        x_uid, x_mid, y_true = batch
        y_pred = self(x_uid, x_mid)
        y_pred_real = self.scaler_inverse_transform(y_pred)
        y_true_real = self.scaler_inverse_transform(y_true)

        return x_uid, x_mid, y_pred, y_true, y_pred_real, y_true_real

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.001)
        trainer = self.trainer
        steps_per_epoch = (
            trainer.limit_train_batches
            if trainer.limit_train_batches > 2
            else len(self.dm.train_dataloader())
        )
        return [optim], [
            {
                "scheduler": OneCycleLR(
                    optim, self.lr,
                    steps_per_epoch=int(steps_per_epoch),
                    epochs=trainer.max_epochs,
                    **self.one_cycle_lr_params
                ),
                "interval": "step",
                "frequency": 1
            },
        ]
