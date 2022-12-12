from typing import Any

import pytorch_lightning as pl
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from opal.score.collaborative_filtering.neu_mf_net import NeuMFNet
from opal.score.collaborative_filtering.utils import adj_inv_sigmoid, adj_sigmoid


class LitNeuMFNet(pl.LightningModule):
    def __init__(self, n_uid, n_mid, mf_emb_dim, mlp_emb_dim, mlp_chn_out,
                 mms_metric: MinMaxScaler, lr: float = 0.0005):
        super().__init__()
        self.model = NeuMFNet(n_uid, n_mid, mf_emb_dim, mlp_emb_dim, mlp_chn_out)
        self.mms_score = mms_metric
        self.lr = lr
        self.loss = MSELoss()

    def forward(self, x_uid, x_mid):
        return self.model(x_uid, x_mid)

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        x_uid, x_mid, y_true = batch
        y_pred_learn = self(x_uid, x_mid)
        y_true_learn = adj_inv_sigmoid(y_true)
        with torch.no_grad():
            y_pred = adj_sigmoid(y_pred_learn)

        # The inv. sigmoid transforms y_true into the learning space to make learning more linear
        # The loss is thus calculated in the learning space.
        loss = self.loss(y_pred_learn, y_true_learn)

        if batch_idx % 32 == 0:
            self.logger.experiment.add_histogram("pred", y_pred)
            self.logger.experiment.add_histogram("true", y_true)
            self.logger.experiment.add_histogram("pred_learn", y_pred_learn)
            self.logger.experiment.add_histogram("true_learn", y_true_learn)
            self.logger.experiment.add_histogram("loss_", y_pred_learn - y_true_learn)

        # The sigmoid transforms predictions from learning to real space
        with torch.no_grad():
            self.log("train_mae",
                     self.mms_score.inverse_transform([[torch.abs(y_pred - y_true).mean().detach().numpy()]])[0, 0])

        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        x_uid, x_mid, y_true = batch
        y_pred_learn = self(x_uid, x_mid)
        y_pred = adj_sigmoid(y_pred_learn)

        self.log("val_mae",
                 self.mms_score.inverse_transform([[torch.abs(y_pred - y_true).mean().detach().numpy()]])[0, 0])

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        x_uid, x_mid, y_true = batch
        return adj_sigmoid(self(x_uid, x_mid)), y_true

    def configure_optimizers(self):
        # optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.001)
        optim = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optim, mode='min', factor=0.2, patience=2, verbose=True),
                "monitor": "val_mae",
            },
        }
