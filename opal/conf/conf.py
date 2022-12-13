from pathlib import Path

ROOT_DIR = Path(__file__).parents[2]
DATA_DIR = ROOT_DIR / "data/"
OSU_DIR = DATA_DIR / "osu/"
MODEL_DIR = ROOT_DIR / "models/"
REPLAYS_DIR = OSU_DIR / "replays"
SCORES_DIR = OSU_DIR / "scores"
