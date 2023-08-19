import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint, StochasticWeightAveraging

from opal.conf import MODEL_DIR, DATASET_DIR
from opal.module import OpalNet
from opal.score_datamodule import ScoreDataModule


def train(experiment_name: str, dataset: str = None):
    """ Trains the OpalNet

    Args:
        experiment_name: Experiment Name Tag. Can be a tag used before, this will append to the experiment.
        dataset: Dataset to train on. If None, the most recent dataset is used.

    Returns:

    """
    if dataset is None:
        # This yields the most recent dataset by modified time
        dataset = sorted(list(DATASET_DIR.glob("*.csv")), key=lambda x: x.stat().st_mtime_ns)[-1].stem

    # The experiment artifacts will be saved to an experiment specific directory in the model directory
    experiment_dir = MODEL_DIR / f"{experiment_name}_{dataset}"
    experiment_dir.mkdir(exist_ok=True, parents=True)

    dm = ScoreDataModule(dataset=dataset, batch_size=2 ** 10)

    epochs = 25
    net = OpalNet(
        uid_le=dm.uid_le,
        mid_le=dm.mid_le,
        transformer=dm.transformer,
        emb_dim=16,
        mlp_range=[1, ],
        lr=1e-3,
    )

    trainer = pl.Trainer(
        deterministic=True,
        max_epochs=epochs,
        accelerator='cpu',
        default_root_dir=experiment_dir,
        log_every_n_steps=50,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=2,
                verbose=True,
                mode='min',
                min_delta=0.0001,
                divergence_threshold=1
            ),
            StochasticWeightAveraging(1e-3),
            LearningRateMonitor(),
            ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')
        ],
    )

    trainer.fit(net, datamodule=dm)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    train("V4")
