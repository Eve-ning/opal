import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint, ModelPruning, QuantizationAwareTraining

from opal.score.collaborative_filtering import NeuMF
from opal.score.datamodule import ScoreDataModule


def train(yyyy_mm: str):
    dm = ScoreDataModule(
        ds_yyyy_mm=yyyy_mm,
        batch_size=2 ** 10,
        score_bounds=(5e5, 1e6),
        ds_set="10000"
    )

    epochs = 50
    net = NeuMF(
        uid_le=dm.uid_le,
        mid_le=dm.mid_le,
        qt=dm.qt_accuracy,
        emb_dim=8,
        mf_repeats=2,
        mlp_range=[128, 64, 32, 8],
        lr=1e-3,
    )

    trainer = pl.Trainer(
        deterministic=True,
        max_epochs=epochs,
        accelerator='gpu',
        default_root_dir="V1_2022_11",
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=3,
                verbose=True,
                mode='min',
                divergence_threshold=1
            ),
            LearningRateMonitor(),
            ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')
        ],
        devices=[3, ],
    )

    trainer.fit(net, datamodule=dm)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    train("2023_01")
