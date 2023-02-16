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
        ds_set="10000"
    )

    epochs = 25
    net = NeuMF(
        uid_le=dm.uid_le,
        mid_le=dm.mid_le,
        qt=dm.qt_accuracy,
        mf_emb_dim=8,
        mlp_emb_dim=8,
        mlp_chn_out=8,
        lr=1e-3,
        # one_cycle_lr_params={
        #     "pct_start": 0.025,
        #     "three_phase": True,
        #     "final_div_factor": 1e7
        # }
    )

    trainer = pl.Trainer(
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
            # StochasticWeightAveraging(0.0005),
            LearningRateMonitor(),
            ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')
        ],
        # strategy=DDPStrategy(find_unused_parameters=False),
        devices=[3, ],
        # fast_dev_run=True,
    )

    trainer.fit(net, datamodule=dm)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    train("2023_01")
