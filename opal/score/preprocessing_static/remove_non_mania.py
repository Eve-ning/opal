from __future__ import annotations

from pathlib import Path

from reamber.osu import OsuMap
from reamber.osu.OsuMapMeta import OsuMapMode
from tqdm import tqdm

from opal.score.dataset import Dataset


def remove_non_mania(ds: Dataset) -> None:
    """ Loops through a directory and removes any maps that aren't mania """
    for file in (t := tqdm(ds.files_path.glob("*.osu"))):
        file: Path
        try:
            m = OsuMap.read_file(file.as_posix())
        except:
            t.set_description(f"Removing {file}")
            file.unlink(missing_ok=False)
            continue
        if m.mode == OsuMapMode.MANIA:
            t.set_description(f"Keeping {file}")
        else:
            t.set_description(f"Removing {file}")
            file.unlink(missing_ok=False)
