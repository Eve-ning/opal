import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint

from opal.conf import ROOT_DIR
from opal.module import OpalNet
from opal.score_datamodule import ScoreDataModule


def train():
    dm = ScoreDataModule(
        dataset="2023_08_01_performance_mania_top_10000_20230818220650.csv",
        batch_size=2 ** 5
    )

    epochs = 3
    net = OpalNet(
        uid_le=dm.uid_le,
        mid_le=dm.mid_le,
        transformer=dm.transformer,
        emb_dim=16,
        lr=1e-3,
    )

    trainer = pl.Trainer(
        deterministic=True,
        max_epochs=epochs,
        accelerator='cpu',
        default_root_dir=ROOT_DIR,
        log_every_n_steps=50,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=1,
                verbose=True,
                mode='min',
                min_delta=0.0001,
                divergence_threshold=1
            ),
            LearningRateMonitor(),
            ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')
        ],
    )

    trainer.fit(net, datamodule=dm)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    train()
