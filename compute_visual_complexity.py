from __future__ import annotations

from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
from reamber.algorithms.analysis import scroll_speed
from reamber.osu import OsuMap
from sqlalchemy import create_engine
from tqdm import tqdm


def osu_map_visual_complexity(m: OsuMap):
    """ Evaluate the visual complexity of a map via integral over the map's visual complexity """

    def visual_complexity(speed: np.ndarray):
        """ Evaluates the VC for the current speed.

        Notes:
            The VC for   1 = 0
                       0-1 = (x-1)^2
                       1-3 = sin((x-2) * 0.5pi) / 2 + 0.5
                             A sigmoid easing from x in [1,3], y goes from 0 to 1.
                        >3 = 1

                ^ VC
                |
            1.0 +xx                 xxxxxxxxxxx
                |  xx            xxx|
                |    xx       xxx   |
           -----+------xx+xxxx------+---------> speed
                |       1.0        3.0
        """
        return np.piecewise(
            speed,
            [
                (0 <= speed) * (speed < 1),
                (1 <= speed) * (speed < 3)
            ],
            [
                lambda x: (x - 1) ** 2,  # x^2 0 to 1
                lambda x: np.sin((x - 2) * np.pi * 0.5) / 2 + 0.5,  # Sigmoid Easing 1 to 3
                1  # Otherwise
            ]
        )

    speed = scroll_speed(m)
    offsets = speed.index

    return np.sum(
        # Evaluate the integral visual complexity w.r.t. time
        (visual_complexity(speed.to_numpy()[:-1]) * np.diff(offsets)) /
        # Take the proportional visual complexity
        (offsets.max() - offsets.min())
    )


def compute_visual_complexity(mids: pd.Series, osu_files_path: Path) -> pd.DataFrame:
    """ Given mids & path to the *.osu files, compute visual complexity of all maps and return as a DataFrame.

    Args:
        mids: pd.Series of the beatmap_ids to compute.
        osu_files_path: Path to the *.osu files.

    Returns:
        A pd.DataFrame of index 'mid' and column 'visual_complexity'.
    """

    # Create the DF we'll populate with vc_ix
    df = pd.DataFrame(dict(mid=mids))

    for mid in tqdm(mids, desc="Evaluating Visual Complexity of Maps..."):
        # Get our osu map
        osu_path = osu_files_path / f"{mid}.osu"
        osu = OsuMap.read_file(osu_path)

        # Get visual_complexity
        vc = osu_map_visual_complexity(osu)

        # Set the Visual Complexity to corresponding map
        df.loc[df['mid'] == mid, 'visual_complexity'] = vc

    return df.set_index('mid')


if __name__ == '__main__':
    db_name: str = "osu"
    user_name: str = "root"
    password: str = "p@ssw0rd1"
    host: str = "opal.mysql"
    port: int = 3307
    quoted_password = quote_plus(password)
    engine = create_engine(f'mysql+mysqlconnector://{user_name}:{quoted_password}@{host}:{port}/{db_name}')

    con = engine.connect()
    mids = pd.read_sql_table("opal_active_mid", con=con)['mid'].unique()
    files_dir = Path("/opal.files") / f"{datetime.now().strftime('%Y_%m')}_01_osu_files/"

    df_vc = compute_visual_complexity(mids, files_dir)
    df_vc.to_sql(name="opal_beatmaps_visual_complexity", con=con, if_exists='replace', )
