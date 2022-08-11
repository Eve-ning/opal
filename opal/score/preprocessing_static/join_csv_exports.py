from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


def join_csv_exports(in_paths: List[str | Path], out_path: str | Path):
    """ Joins multiple CSV exports together an exports it

    Notes:
        This usually only happens if the Google Cloud doesn't support >1GB
        exports.

    Args:
        in_paths: Input paths of the CSV exports
        out_path: Output path of the joined export

    """

    dfs = []
    for in_path in in_paths:
        dfs.append(pd.read_csv(in_path))

    df = pd.concat(dfs)
    df.to_csv(Path(out_path), index=False)


if __name__ == '__main__':
    in_paths = [
        Path("../../data/exportcsv000000000000.csv"),
        Path("../../data/exportcsv000000000001.csv"),
        Path("../../data/exportcsv000000000002.csv"),
        Path("../../data/exportcsv000000000003.csv"),
        Path("../../data/exportcsv000000000004.csv"),
        Path("../../data/exportcsv000000000005.csv"),
        Path("../../data/exportcsv000000000006.csv"),
        Path("../../data/exportcsv000000000007.csv"),
        Path("../../data/exportcsv000000000008.csv")
    ]

    join_csv_exports(in_paths,
                     "../../data/osu/2022_04_top_1000/scores_top10k.csv")
