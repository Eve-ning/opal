from pathlib import Path

PKG_DIR = Path(__file__).parents[1]
ROOT_DIR = PKG_DIR.parent

DATA_DIR = ROOT_DIR / "data/"
MODEL_DIR = PKG_DIR / "models/"

MODEL_CKPT = MODEL_DIR / "V3_2023_04/model.ckpt"
