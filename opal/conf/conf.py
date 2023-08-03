from datetime import datetime
from pathlib import Path

# PROJ_DIR is opal/, PKG_DIR is opal/opal/
PROJ_DIR = Path(__file__).parents[2]
PKG_DIR = PROJ_DIR / "opal"

MODEL_DIR = PKG_DIR / "models/"
LATEST_MODEL_CKPT = MODEL_DIR / "V3_2023_05/model.ckpt"


def get_current_files_dir() -> Path:
    """ Gets the current local docker mounted osu.files directory """
    files_dir = PROJ_DIR / "opal.files" / f"{datetime.now().strftime('%Y_%m')}_01_osu_files/"
    assert files_dir.is_dir(), f"Failed to get files dir, {files_dir.as_posix()} is not a directory."
    return files_dir
