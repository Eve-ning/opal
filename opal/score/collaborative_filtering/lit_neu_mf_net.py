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
        x, y = batch
        y_hat = self(x)
        y_adj = adj_inv_sigmoid(y)

        # We use an inv. sigmoid to make the model learn from a more linear accuracy curve.
        # Here, we measure the loss of the pred - linearized acc
        loss = torch.sqrt(((y_hat - y_adj) ** 2).mean())

        # As we learnt the linearized acc, we need to transform it back to something interpretable
        # We sigmoid it to make it the actual curved acc.
        self.log("train_mae", torch.abs(adj_sigmoid(y_hat) - adj_sigmoid(y_adj)).mean())

        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch

        y_hat = self(x)
        self.log("val_mae", torch.abs(adj_sigmoid(y_hat) - y).mean())

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
