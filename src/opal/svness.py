from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
from reamber.algorithms.analysis import scroll_speed
from reamber.osu import OsuMap
from sqlalchemy import create_engine
from tqdm import tqdm


def compute_map_svness(m: OsuMap):
    """ Evaluate the 'SV'-ness of a map via integral over the map's deviation from 1.0 """

    def compute_speed_svness(speed: np.ndarray):
        """ Evaluates the deviation for the current speed.

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
        # Evaluate the integral svness w.r.t. time
        (compute_speed_svness(speed.to_numpy()[:-1]) * np.diff(offsets)) /
        # Take the proportional svness
        (offsets.max() - offsets.min())
    )


def compute_maps_svness(mids: pd.Series, osu_files_path: Path) -> pd.DataFrame:
    """ Given mids & path to the *.osu files, compute svness of all maps and return as a DataFrame.

    Args:
        mids: pd.Series of the beatmap_ids to compute.
        osu_files_path: Path to the *.osu files.

    Returns:
        A pd.DataFrame of index 'mid' and column 'visual_complexity'.
    """

    # Create the DF we'll populate with vc_ix
    df = pd.DataFrame(dict(mid=mids))
    df.loc[:, 'svness'] = None
    for mid in tqdm(mids, desc="Evaluating SV-ness of Maps..."):
        # Get our osu map
        osu_path = osu_files_path / f"{mid}.osu"
        # TODO: I think this might not be the best way to do this.
        if not osu_path.exists():
            print(f"Map {mid} does not exist in {osu_files_path}")
            continue
        osu = OsuMap.read_file(osu_path)

        # Get visual_complexity
        vc = compute_map_svness(osu)

        # Set the svness to corresponding map
        df.loc[df['mid'] == mid, 'svness'] = vc

    return df.set_index('mid')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='svness.py', description='Evalaute SV-Ness of maps')
    parser.add_argument('--files_path', type=str, help='Path to *.osu files')
    parser.add_argument('--db_name', type=str, help='Database Name')
    parser.add_argument('--db_username', type=str, help='Database User Name')
    parser.add_argument('--db_password', type=str, help='Database Password')
    parser.add_argument('--db_host', type=str, help='Database Host')
    parser.add_argument('--db_port', type=str, help='Database Port')
    args = parser.parse_args()

    if not args.files_path:
        parser.print_help()
        sys.exit(1)

    files_path: Path = Path(args.files_path)

    db_name = args.db_name if args.db_name else "osu"
    db_username = args.db_username if args.db_username else "root"
    db_password = args.db_password if args.db_password else "p@ssw0rd1"
    db_host = args.db_host if args.db_host else "osu.mysql"
    db_port = args.db_port if args.db_port else 3307

    engine = create_engine(
        f'mysql+mysqlconnector://'
        f'{db_username}:{quote_plus(db_password)}@{db_host}:{db_port}/{db_name}'
    )

    con = engine.connect()
    mids = pd.read_sql_table("opal_active_mid", con=con)['mid'].unique()

    df_vc = compute_maps_svness(mids, files_path)
    df_vc.to_sql(name="opal_active_mid_svness", con=con, if_exists='replace', )
