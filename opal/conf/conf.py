from datetime import datetime
from pathlib import Path

# PROJ_DIR is opal/, PKG_DIR is opal/opal/
PROJ_DIR = Path(__file__).parents[2]
PKG_DIR = PROJ_DIR / "opal"

MODEL_DIR = PKG_DIR / "models/"
LATEST_MODEL_CKPT = MODEL_DIR / "V3_2023_05/model.ckpt"
