import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint, StochasticWeightAveraging

from opal.conf import MODEL_DIR, DATASET_DIR
from opal.module import OpalNet
from opal.score_datamodule import ScoreDataModule


def train(model_name: str, dataset_path: Path, pipeline_run_cache: Path = None):
    """ Trains the OpalNet

    Args:
        model_name: Experiment Name Tag. Can be a tag used before, this will append to the experiment.
        dataset_path: Path to the dataset.

    """

    # The experiment artifacts will be saved to an experiment specific directory in the model directory
    # TODO: Differentiate between MODEL_DIR and model_dir
    model_dir = MODEL_DIR / model_name / dataset_path.name
    model_dir.mkdir(exist_ok=True, parents=True)

    dm = ScoreDataModule(dataset_path=dataset_path, batch_size=2 ** 10)

    epochs = 25
    net = OpalNet(
        uid_le=dm.uid_le,
        mid_le=dm.mid_le,
        transformer=dm.transformer,
        emb_dim=8,
        lr=1e-3,
    )

    trainer = pl.Trainer(
        deterministic=True,
        max_epochs=epochs,
        accelerator='cpu',
        default_root_dir=model_dir,
        log_every_n_steps=50,
        limit_train_batches=3,
        limit_val_batches=3,
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
            ModelCheckpoint(
                monitor='val_loss', save_top_k=1, mode='min'
            )
        ],
    )

    trainer.fit(net, datamodule=dm)

    # This is only used for pipeline runs
    if pipeline_run_cache:
        model_path = Path(trainer.checkpoint_callback.best_model_path)
        with open(pipeline_run_cache, 'a') as f:
            f.write(f'MODEL_PATH={model_path}\n')


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    parser = argparse.ArgumentParser(prog='train.py', description='Train OpalNet')
    parser.add_argument('--model_name', type=str,
                        help='Experiment Name Tag. Can be a tag used before, this will append to the experiment dir.')
    parser.add_argument('--dataset_name', type=str,
                        help='Dataset Name, must be in ../datasets/<DATASET_NAME>')
    parser.add_argument('--pipeline_run_cache', type=str,
                        help='Path to the pipeline run cache file. Optional, used for pipeline runs.')
    args = parser.parse_args()

    if not (args.pipeline_run_cache or args.model_name or args.dataset_name):
        parser.print_help()
        sys.exit(1)

    MODEL_NAME = args.model_name
    DATASET_PATH = DATASET_DIR / args.dataset_name
    PIPELINE_RUN_CACHE = Path(args.pipeline_run_cache) if args.pipeline_run_cache else None
    assert MODEL_NAME, "Model Name must be provided."
    assert DATASET_PATH.exists(), f"Dataset {DATASET_PATH.as_posix()} does not exist."

    train(MODEL_NAME, DATASET_PATH, PIPELINE_RUN_CACHE)
