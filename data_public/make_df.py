from pathlib import Path

import pandas as pd
from tqdm import tqdm

from opal.replay.preprocess_replay_error import preprocess_replay_error


def make_df(path: Path, pkl_name: str):
    map_dirs = [d for d in path.iterdir() if d.is_dir()]
    dfs = []
    for d in tqdm(map_dirs):
        df = preprocess_replay_error(d)
        dfs.append(df)
    df = pd.concat(dfs)

    df: pd.DataFrame
    df.to_pickle(f"{pkl_name}.pkl")


if __name__ == '__main__':
    make_df(Path("osu/train_test/"), "train_test.pkl")
    make_df(Path("osu/validation/"), "validation.pkl")
