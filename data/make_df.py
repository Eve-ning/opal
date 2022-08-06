from pathlib import Path

import pandas as pd
from tqdm import tqdm

from opal.conf import DATA_DIR
from opal.utils import load_map


def make_df(path: Path, pkl_name: str):
    map_dirs = [d for d in path.iterdir() if d.is_dir()]
    dfs_err = []
    for d in tqdm(map_dirs):
        df_err, osu = load_map(d)
        dfs_err.append(df_err)
    df = pd.concat(dfs_err)
    df.to_pickle(pkl_name)


if __name__ == '__main__':
    make_df(DATA_DIR, "osu.pkl")
