import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from opal.score.collaborative_filtering.neu_mf_net import NeuMFNet
from opal.score.collaborative_filtering.utils import adj_inv_sigmoid, adj_sigmoid


class LitNet(pl.LightningModule):
    def __init__(self, uid_no, mid_no, mf_emb_dim, mlp_emb_dim, mlp_chn_out, scaler: TransformerMixin):
        super().__init__()
        self.model = NeuMFNet(uid_no, mid_no, mf_emb_dim, mlp_emb_dim, mlp_chn_out)
        self.loss = MSELoss()
        self.scaler = scaler

    def forward(self, uid, mid):
        return self.model(uid, mid)

    def scaler_inverse(self, val: torch.Tensor):
        return self.scaler.inverse_transform(val.detach().numpy())

    def training_step(self, batch, batch_idx):
        x_uid, x_mid, y = batch
        y_hat = self(x_uid, x_mid)
        # y_adj = adj_inv_sigmoid(y)

        if batch_idx % 32 == 0:
            self.logger.experiment.add_histogram("pred", y_hat)
            self.logger.experiment.add_histogram("true", y)
            # self.logger.experiment.add_histogram("pred_learn", y_hat)
            # self.logger.experiment.add_histogram("true_learn", y_adj)

        loss = self.loss(y_hat, y)
        self.log(
            "train_mae",
            np.abs(
                self.scaler_inverse(y_hat) - self.scaler_inverse(y)
            ).mean()
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x_uid, x_mid, y = batch
        y_hat = self(x_uid, x_mid)
        # y_adj = adj_inv_sigmoid(y)
        self.log(
            "val_mae",
            np.abs(
                self.scaler_inverse(y_hat) - self.scaler_inverse(y)
            ).mean()
        )

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=0.001)

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optim, mode='min', factor=0.2, patience=10, verbose=True),
                "monitor": "train_mae",
            },
        }
