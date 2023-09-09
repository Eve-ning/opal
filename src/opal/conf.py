from pathlib import Path

ROOT_DIR = Path(__file__).parents[1]
DATASET_DIR = ROOT_DIR / "datasets"
MODEL_DIR = ROOT_DIR / "opal/models"
EXPERIMENT_DIR = ROOT_DIR / "lightning_logs"

LATEST_MODEL_CKPT = (
        MODEL_DIR /
        "2023.9.5b/2023_09_01_performance_mania_top_10000_44e44645_84947aba.csv/lightning_logs/version_2/evaluation"
        "/model.ckpt"
)
