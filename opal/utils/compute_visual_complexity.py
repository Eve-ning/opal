from __future__ import annotations

from pathlib import Path

import docker
import numpy as np
import pandas as pd
from reamber.algorithms.analysis import scroll_speed
from reamber.osu import OsuMap
from tqdm import tqdm

client = docker.from_env()
from sqlalchemy import create_engine



def create_visual_complexity_table(osu_files_path: Path) -> pd.DataFrame:
    mids = pd.read_sql_table("opal_active_mid", con=con)['mid'].unique()

    # Create the DF we'll populate with vc_ix
    df = pd.DataFrame(dict(mid=mids))

    def visual_complexity(x):
        # Visual Complexity Equation
        # Below 1.0, we use a simple x^2
        # Above 1.0, we do a sigmoid easing from 1 to 3
        # Otherwise, we set to 1.0
        return np.piecewise(
            x,
            [
                (0 <= x) * (x < 1),
                (1 <= x) * (x < 3)
            ],
            [
                lambda x: (x - 1) ** 2,  # x^2 0 to 1
                lambda x: np.sin((x - 2) * np.pi * 0.5) / 2 + 0.5,  # Sigmoid Easing 1 to 3
                1  # Otherwise
            ]
        )

    def osu_visual_complexity(osu):
        speed = scroll_speed(osu)
        offsets = speed.index

        return np.sum(
            # Evaluate the integral visual complexity w.r.t. time
            (visual_complexity(speed.to_numpy()[:-1]) * np.diff(offsets)) /
            # Take the proportional visual complexity
            (offsets.max() - offsets.min())
        )

    for mid in tqdm(mids, desc="Evaluating Visual Complexity of Maps..."):
        # Get our osu map
        osu_path = osu_files_path / f"{mid}.osu"
        osu = OsuMap.read_file(osu_path)

        # Get visual_complexity
        vc = osu_visual_complexity(osu)

        # Set the Visual Complexity to corresponding map
        df.loc[df['mid'] == mid, 'visual_complexity'] = vc

    df = df.set_index('mid')

    # Send to sql
    df.to_sql(
        name="opal_beatmaps_visual_complexity",
        con=self.mysql_engine,
        if_exists='replace',
    )
    con.close()
    return df
