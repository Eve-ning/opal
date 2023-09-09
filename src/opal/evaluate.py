""" This script is used to evaluate the model. It is not used in the training process. """
import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
from sklearn.metrics import r2_score

from opal.conf import DATASET_DIR
from opal.module import OpalNet
from opal.score_datamodule import ScoreDataModule


def compute_test_predictions(dataset_path: Path, model_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """ Computes the predictions for the test set """
    dm = ScoreDataModule(batch_size=16, dataset_path=dataset_path)
    net = OpalNet.load_from_checkpoint(model_path)
    net.eval()
    trainer = pl.Trainer(accelerator='cpu', logger=False)
    y = trainer.predict(net, datamodule=dm)
    y_preds = []
    y_trues = []
    # The -1 is to avoid stacking jagged arrays
    for x_uid_real, x_mid_real, y_pred, y_true in y[:-1]:
        y_preds.append(y_pred)
        y_trues.append(y_true)
    return np.stack(y_preds).flatten(), np.stack(y_trues).flatten()


def plot_overview(y_preds: np.ndarray, y_trues: np.ndarray, dataset_path: Path, save_path: Path):
    """ Plots an overview of the predictions and actual accuracies """
    r2 = r2_score(y_preds, y_trues)
    mae = np.abs(y_preds - y_trues).mean()
    rmse = ((y_preds - y_trues) ** 2).mean() ** 0.5
    y_preds = y_preds[:10000]
    y_trues = y_trues[:10000]

    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(10, 10))
    plt.suptitle(
        f"Dataset: {dataset_path}\n"
        f"{r2=:.2%} {mae=:.2%} {rmse=:.2%}\n"
        f"Overview of Predictions and Actual Accuracies"
    )

    ax1 = plt.subplot(221)
    sns.lineplot(x=[0.85, 1], y=[0.85, 1], color='gray')
    sns.scatterplot(x=y_trues, y=y_preds, s=8, c=np.abs(y_trues - y_preds),
                    cmap='magma')
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0, xmax=1))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0, xmax=1))
    plt.title("Predictions against Actual")
    plt.xlabel("Actual Accuracy")
    plt.ylabel("Predicted Accuracy")

    ax2 = plt.subplot(222)
    ax2.sharex(ax1)
    sns.histplot(x=y_trues, y=np.abs(y_preds - y_trues), bins=25)
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0, xmax=1))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0, xmax=1))
    plt.title("Prediction Error Heatmap")
    plt.xlabel("Actual Accuracy")
    plt.ylabel("Absolute Accuracy Prediction Error")

    ax3 = plt.subplot(223)
    ax3.sharex(ax1)
    sns.histplot(x=y_preds, bins=25)
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0, xmax=1))
    plt.title("Prediction Distribution")
    plt.xlabel("Predicted Accuracy")
    plt.ylabel("Frequency of Prediction")

    ax4 = plt.subplot(224)
    ax4.sharex(ax1)
    sns.histplot(x=y_trues, bins=25)
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0, xmax=1))
    plt.title("Actual Distribution")
    plt.xlabel("Actual Accuracy")
    plt.ylabel("Frequency of Actual")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_error_distribution(y_preds: np.ndarray, y_trues: np.ndarray, save_path: Path):
    """ Plots the error distribution """

    def get_error(y_preds, y_trues, a, b):
        y_preds = y_preds[(y_trues >= a) & (y_trues < b)]
        y_trues = y_trues[(y_trues >= a) & (y_trues < b)]

        return np.abs(y_preds - y_trues).mean(), ((y_preds - y_trues) ** 2).mean() ** 0.5

    errors = []
    bounds = np.linspace(0.7, 1, 31)
    for a, b in zip(bounds[:-1], bounds[1:]):
        mae, rmse = get_error(y_preds, y_trues, a, b)
        errors.append([f"{b:.0%}", mae, rmse])

    df_errors = pd.DataFrame(errors, columns=['bounds', 'mae', 'rmse'])
    plt.figure(figsize=(14, 4))
    sns.lineplot(data=df_errors, x='bounds', y='mae', label='mae')
    sns.lineplot(data=df_errors, x='bounds', y='rmse', label='rmse')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

    plt.title("Error Distribution")
    plt.xlabel("Actual Accuracies")
    plt.ylabel("Error")
    _ = plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='evaluate.py', description='Evaluate OpalNet Model')
    parser.add_argument('--model_path', type=str,
                        help='Path to the model checkpoint')
    parser.add_argument('--dataset_name', type=str,
                        help='Dataset Name, must be in ../datasets/<DATASET_NAME>')
    args = parser.parse_args()

    MODEL_PATH = Path(args.model_path)
    DATASET_PATH = DATASET_DIR / args.dataset_name
    assert MODEL_PATH.exists(), f"Model {MODEL_PATH.as_posix()} does not exist."
    assert DATASET_PATH.exists(), f"Dataset {DATASET_PATH.as_posix()} does not exist."

    # Model Path: /app/opal/models/experiment_name/dataset/lightning_logs/version_X/checkpoints/___.ckpt
    eval_dir = MODEL_PATH.parents[1] / "evaluation"
    eval_dir.mkdir(exist_ok=True)

    y_preds, y_trues = compute_test_predictions(dataset_path=DATASET_PATH,
                                                model_path=MODEL_PATH)
    shutil.copy(MODEL_PATH, eval_dir / "model.ckpt")
    plot_overview(y_preds, y_trues, DATASET_PATH, eval_dir / "overview.png")
    plot_error_distribution(y_preds, y_trues, eval_dir / "error_distribution.png")
