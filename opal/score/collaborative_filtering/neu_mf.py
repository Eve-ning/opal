import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import QuantileTransformer
from torch.nn import MSELoss
from torch.optim.lr_scheduler import StepLR

from opal.score.collaborative_filtering.neu_mf_module import NeuMFModule


class NeuMF(pl.LightningModule):
    def __init__(self, uid_no, mid_no, mf_emb_dim, mlp_emb_dim, mlp_chn_out, scaler: QuantileTransformer,
                 lr: float = 0.005):
        super().__init__()
        self.model = NeuMFModule(uid_no, mid_no, mf_emb_dim, mlp_emb_dim, mlp_chn_out)
        self.loss = MSELoss()
        self.scaler = scaler
        self.lr = lr
        self.save_hyperparameters(ignore=['scaler'])

    def forward(self, uid, mid):
        return self.model(uid, mid)

    def scaler_inverse(self, val: torch.Tensor):
        return self.dm.scaler_accuracy.inverse_transform(val.detach().numpy())

    def uid_inverse(self, val: torch.Tensor):
        return self.dm.uid_le.inverse_transform(val.detach().numpy())

    def mid_inverse(self, val: torch.Tensor):
        return self.dm.mid_le.inverse_transform(val.detach().numpy())

    def training_step(self, batch, batch_idx):
        x_uid, x_mid, y_pred, y_true, y_pred_real, y_true_real = self.step(batch)

        # if batch_idx % 32 == 0:
        #     self.logger.experiment.add_histogram("pred", y_pred)
        #     self.logger.experiment.add_histogram("true", y_true)

        loss = self.loss(y_pred, y_true)
        self.log("Train MAE (%)", np.abs(y_pred_real - y_true_real).mean(), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        *_, y_pred_real, y_true_real = self.step(batch)
        self.log("Val MAE (%)", np.abs(y_pred_real - y_true_real).mean(), prog_bar=True)

    def predict_step(self, batch, batch_idx, **kwargs):
        x_uid, x_mid, *_, y_pred_real, y_true_real = self.step(batch)

        x_uid_real = self.uid_inverse(x_uid)
        x_mid_real = self.mid_inverse(x_mid)

        return x_uid_real, x_mid_real, y_pred_real, y_true_real

    def step(self, batch):
        x_uid, x_mid, y_true = batch
        y_pred = self(x_uid, x_mid)
        y_pred_real = self.scaler_inverse(y_pred)
        y_true_real = self.scaler_inverse(y_true)

        return x_uid, x_mid, y_pred, y_true, y_pred_real, y_true_real

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
