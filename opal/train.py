import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint

from opal.score.collaborative_filtering import NeuMF
from opal.score.datamodule import ScoreDataModule


def train(yyyy_mm: str):
    dm = ScoreDataModule(
        ds_yyyy_mm=yyyy_mm,
        batch_size=2 ** 10,
        score_bounds=(5e5, 1e6),
        ds_set="1000"
    )

    epochs = 50
    net = NeuMF(
        uid_le=dm.uid_le,
        mid_le=dm.mid_le,
        qt=dm.qt_accuracy,
        emb_dim=128,
        mlp_range=[512, 256, 128, 64, 32, 32],
        lr=1e-3,
        # one_cycle_lr_params={
        #     "pct_start": 0.025,
        #     "three_phase": True,
        #     "final_div_factor": 1e7
        # }
    )

    trainer = pl.Trainer(
        deterministic=True,
        max_epochs=epochs,
        accelerator='gpu',
        default_root_dir="V1_2022_11",
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=20,
                verbose=True,
                mode='min',
                divergence_threshold=1
            ),
            LearningRateMonitor(),
            ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')
        ],
        devices=[3, ],
        # fast_dev_run=True,
    )

    trainer.fit(net, datamodule=dm)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    train("2023_01")
