from pathlib import Path
import platform

if platform.system() == 'Windows':
    ROOT_DIR = Path(__file__).parents[1]
else:
    ROOT_DIR = Path("/var/lib/opal/")
DATASET_DIR = ROOT_DIR / "datasets"
MODEL_DIR = ROOT_DIR / "models"
EXPERIMENT_DIR = ROOT_DIR / "lightning_logs"

LATEST_MODEL_CKPT = MODEL_DIR / "V3_2023_05/model.ckpt"
