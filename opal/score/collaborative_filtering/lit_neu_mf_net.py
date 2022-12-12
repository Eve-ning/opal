from typing import Any

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from opal.score.collaborative_filtering.neu_mf_net import NeuMFNet
from opal.score.collaborative_filtering.utils import adj_inv_sigmoid, adj_sigmoid


class LitNeuMFNet(pl.LightningModule):
    def __init__(self, n_uid, n_mid, mf_emb_dim, mlp_emb_dim, mlp_chn_out):
        super().__init__()
        self.model = NeuMFNet(n_uid, n_mid, mf_emb_dim, mlp_emb_dim, mlp_chn_out)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        x, y_true = batch
        y_pred_learn = self(x)
        y_pred = adj_sigmoid(y_pred_learn)
        y_true_learn = adj_inv_sigmoid(y_true)

        # The inv. sigmoid transforms y_true into the learning space to make learning more linear
        # The loss is thus calculated in the learning space.
        loss = torch.sqrt(((y_pred_learn - y_true_learn) ** 2).mean())

        # The sigmoid transforms predictions from learning to real space
        self.log("train_mae", torch.abs(y_pred - y_true).mean())

        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        x, y_true = batch
        y_pred_learn = self(x)
        y_pred = adj_sigmoid(y_pred_learn)
        self.log("val_mae", torch.abs(y_pred - y_true).mean())

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        x, y = batch
        return adj_sigmoid(self(x))

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.0005, weight_decay=0.001)

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optim, mode='min', factor=0.2, patience=2, verbose=True),
                "monitor": "val_mae",
            },
        }
