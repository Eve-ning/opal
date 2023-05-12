import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint, StochasticWeightAveraging

from opal.datamodule import ScoreDataModule
from opal.module import OpalNet


def train(version: str):
    dm = ScoreDataModule(
        batch_size=2 ** 7,
        accuracy_bounds=(0.85, 1.0)
    )

    epochs = 50
    net = OpalNet(
        uid_le=dm.uid_le,
        mid_le=dm.mid_le,
        transformer=dm.transformer,
        emb_dim=8,
        mf_repeats=3,
        mlp_range=[128, 64, 32, 8],
        lr=1e-3,
        lr_gamma=0.75,
    )

    trainer = pl.Trainer(
        deterministic=True,
        max_epochs=epochs,
        accelerator='gpu',
        default_root_dir=version,

        callbacks=[
            StochasticWeightAveraging(swa_lrs=1e-2),
            EarlyStopping(
                monitor="val_loss",
                patience=3,
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
    train("V3_2023_04")
