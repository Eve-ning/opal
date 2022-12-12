import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from opal.score.collaborative_filtering.neu_mf_net import NeuMFNet
from opal.score.collaborative_filtering.utils import adj_inv_sigmoid, adj_sigmoid


class LitNet(pl.LightningModule):
    def __init__(self, uid_no, mid_no, mf_emb_dim, mlp_emb_dim, mlp_chn_out):
        super().__init__()
        self.model = NeuMFNet(uid_no, mid_no, mf_emb_dim, mlp_emb_dim, mlp_chn_out)

    def forward(self, uid, mid):
        return self.model(uid, mid)

    def training_step(self, batch, batch_idx):
        x_uid, x_mid, y = batch
        y_hat = self(x_uid, x_mid)
        y_adj = adj_inv_sigmoid(y)

        if batch_idx % 32 == 0:
            self.logger.experiment.add_histogram("pred", adj_sigmoid(y_hat))
            self.logger.experiment.add_histogram("true", y)
            self.logger.experiment.add_histogram("pred_learn", y_hat)
            self.logger.experiment.add_histogram("true_learn", y)

        loss = torch.sqrt(((y_hat - y_adj) ** 2).mean())
        self.log("train_mae", torch.abs(adj_sigmoid(y_hat) - adj_sigmoid(y_adj)).mean())

        return loss

    def validation_step(self, batch, batch_idx):
        x_uid, x_mid, y = batch
        y_hat = self(x_uid, x_mid)
        y_adj = adj_inv_sigmoid(y)
        # self.log("val_rmse", adj_sigmoid(y_del ** 2).mean())
        self.log("val_mae", torch.abs(adj_sigmoid(y_hat) - adj_sigmoid(y_adj)).mean())

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.0005, weight_decay=0.001)

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optim, mode='min', factor=0.2, patience=2, verbose=True),
                "monitor": "val_mae",
            },
        }
