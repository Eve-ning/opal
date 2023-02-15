import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy

from opal.score.collaborative_filtering import NeuMF
from opal.score.datamodule import ScoreDataModule


def train(yyyy_mm: str):
    dm = ScoreDataModule(
        ds_yyyy_mm=yyyy_mm,
        batch_size=2 ** 9,
        score_bounds=(5e5, 1e6),
    )

    net = NeuMF(
        uid_le=dm.uid_le,
        mid_le=dm.mid_le,
        qt=dm.qt_accuracy,
        mf_emb_dim=16,
        mlp_emb_dim=16,
        mlp_chn_out=8,
        lr=0.005,
        one_cycle_lr_params={
            "pct_start": 0.1,
            "three_phase": True,
            "final_div_factor": 1e6
        }
    )

    trainer = pl.Trainer(
        max_epochs=35,
        accelerator='gpu',
        default_root_dir="V1_2022_11",
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=2,
                verbose=True,
                mode='min',
                divergence_threshold=1
            ),
            LearningRateMonitor()
        ],
        strategy=DDPStrategy(find_unused_parameters=False),
        # devices=[3, ],
        fast_dev_run=True,
    )

    trainer.fit(net, datamodule=dm)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    train("2023_01")
