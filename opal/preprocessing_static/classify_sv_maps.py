from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from reamber.osu import OsuMap
from tqdm import tqdm

from opal.dataset import Dataset

ACCEPTABLE_SV_MIN = 0.95
ACCEPTABLE_SV_MAX = 1.05


def classify_sv_maps(ds: Dataset) -> None:
    """ Sorts maps in the dir to SV and Non SV Maps """

    def check_sv_map(m: OsuMap, min_sv: float, max_sv: float) -> bool:
        """ Checks if a map is a SV map

        Args:
            m: Osu Map
            min_sv: Minimum Acceptable SV
            max_sv: Maximum Acceptable SV

        Returns:
            Whether if the map has significant SVs or not
        """
        # The only time there's speed change is if there's sv or > 1 bpm
        if len(m.svs) != 0 or len(m.bpms) > 1:

            # We merge both to find associated BPM to SVs
            m_join = pd.merge(m.bpms.df, m.svs.df, how='outer') \
                .sort_values('offset') \
                .ffill().bfill()

            # If there's 1 SV and 1 BPM on the same spot, we can't diff
            if len(m_join) == 1:
                common_bpm = m_join.bpm[0]
            else:
                common_bpm = m_join.iloc[np.argmax(np.diff(m_join.offset))].bpm
            speed = (m_join.bpm * m_join.multiplier) / common_bpm
        else:
            return False

        # If there are any speeds outside the bounds
        return ((speed < min_sv) | (speed > max_sv)).any()

    # Make directories if not done yet.
    ds.sv_maps_path.mkdir(parents=True, exist_ok=True)
    ds.nsv_maps_path.mkdir(parents=True, exist_ok=True)

    for file in (t := tqdm(ds.files_path.glob("*.osu"))):
        file: Path
        m = OsuMap.read_file(file.as_posix())
        if check_sv_map(m, ACCEPTABLE_SV_MIN, ACCEPTABLE_SV_MAX):
            t.set_description(f"Moving SV Map {m.metadata()}")
            target = ds.sv_maps_path / file.name
        else:
            t.set_description(f"Non SV Map {m.metadata()}")
            target = ds.nsv_maps_path / file.name
        shutil.move(file, target)
