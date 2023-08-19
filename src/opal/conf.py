from pathlib import Path
import platform

if platform.system() == 'Windows':
    ROOT_DIR = Path("../../")
else:
    ROOT_DIR = Path("/var/lib/opal/")
DATASET_DIR = ROOT_DIR / "datasets"
MODEL_DIR = ROOT_DIR / "models"

LATEST_MODEL_CKPT = MODEL_DIR / "V3_2023_05/model.ckpt"
