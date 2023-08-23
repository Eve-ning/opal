from pathlib import Path

ROOT_DIR = Path(__file__).parents[1]
DATASET_DIR = ROOT_DIR / "datasets"
MODEL_DIR = ROOT_DIR / "opal/models"
EXPERIMENT_DIR = ROOT_DIR / "lightning_logs"

LATEST_MODEL_CKPT = (
        MODEL_DIR /
        "V4/2023_08_01_performance_mania_top_10000_20230819163602.csv/"
        "lightning_logs/version_1/evaluation/model.ckpt"
)
