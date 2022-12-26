import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.base import TransformerMixin
from sklearn.preprocessing import QuantileTransformer
from torch.nn import MSELoss
from torch.optim.lr_scheduler import StepLR

from opal.score.collaborative_filtering.neu_mf_net import NeuMFNet


class LitNeuMFNet(pl.LightningModule):
    def __init__(self, uid_no, mid_no, mf_emb_dim, mlp_emb_dim, mlp_chn_out, scaler: QuantileTransformer,
                 lr: float = 0.005):
        super().__init__()
        self.model = NeuMFNet(uid_no, mid_no, mf_emb_dim, mlp_emb_dim, mlp_chn_out)
        self.loss = MSELoss()
        self.scaler = scaler
        self.lr = lr
        self.save_hyperparameters(ignore=['scaler'])

    def forward(self, uid, mid):
        return self.model(uid, mid)

    def scaler_inverse(self, val: torch.Tensor):
        return self.scaler.inverse_transform(val.detach().numpy())

    def training_step(self, batch, batch_idx):
        x_uid, x_mid, y = batch
        y_hat = self(x_uid, x_mid)

        if batch_idx % 32 == 0:
            self.logger.experiment.add_histogram("pred", y_hat)
            self.logger.experiment.add_histogram("true", y)

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
        self.log(
            "val_mae",
            np.abs(
                self.scaler_inverse(y_hat) - self.scaler_inverse(y)
            ).mean()
        )

    def predict_step(self, batch, batch_idx, **kwargs):
        x_uid, x_mid, y = batch
        return self.scaler_inverse(self(x_uid, x_mid)), self.scaler_inverse(y)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.001)

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": StepLR(optim, step_size=1, gamma=0.2),
                # "scheduler": ReduceLROnPlateau(optim, mode='min', factor=0.2, patience=10, verbose=True),
                # "monitor": "train_mae",
            },
        }
