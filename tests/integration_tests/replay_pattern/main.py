from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from reamber.algorithms.osu.OsuReplayError import osu_replay_error
from reamber.algorithms.pattern import Pattern
from reamber.algorithms.pattern.combos import PtnCombo
from reamber.osu.OsuMap import OsuMap
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, \
    GradientBoostingRegressor
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler

from opal.conf.conf import OSU_REPLAYS_DIR

def get_errors(rep_paths: List[str], osu: OsuMap) -> pd.DataFrame:
    """ Gets the errors of the replays.

    Args:
        rep_paths: Paths of the replays
        osu: Path of the osu! map

    """
    errors = osu_replay_error(rep_paths, osu)
    rep_count = len(rep_paths)
    df_map_offset = pd.DataFrame.from_records(
        [(v_,)
         for k, v in [*errors.map_offsets.hits.items(),
                      *errors.map_offsets.releases.items()]
         for v_ in v],
        columns=["offset"]
    )
    df_errors = pd.DataFrame.from_records(
        [(r_id, k, v_)
         for r_id, rep_offset in enumerate(errors.errors)
         for k, v in [*rep_offset.hits.items(),
                      *rep_offset.releases.items()]
         for v_ in v],
        columns=["r_id", "column", "error"]
    )
    return pd.merge(
        pd.concat([df_map_offset] * rep_count).reset_index(),
        df_errors,
        left_index=True,
        right_index=True
    ).drop('index', axis=1).astype(int)


def get_pattern(osu: OsuMap) -> pd.DataFrame:
    """ Gets the pattern of the map.

    Args:
        osu: Path of the osu! map
    """
    groups = Pattern.from_note_lists([osu.hits, osu.holds], False).group()
    combos = np.concatenate(PtnCombo(groups).combinations(size=2), axis=0)
    return pd.DataFrame({
        "offset": combos['offset'][:, 0],
        "column": combos['column'][:, 0],
        "column_next": combos['column'][:, 1],
        "diff": np.diff(combos['offset'], axis=1).squeeze(),
    }).astype(int)


def get_comparison(map_dir: Path) -> pd.DataFrame:
    """ Gets the replay-pattern comparison

    Args:
        map_dir: Path of the osu! map directory
    """
    map_path = map_dir / (map_dir.name + ".osu")
    osu = OsuMap.read_file(map_path)
    rep_dir = map_dir / "rep"
    rep_paths = [p.as_posix() for p in rep_dir.iterdir() if p.is_file()]

    return pd.merge(get_pattern(osu), get_errors(rep_paths, osu),
                    how='left', on=['column', 'offset'])


def make_dummies(df: pd.DataFrame):
    """ Convert column variables to dummies

    Args:
        df: DataFrame with column variables
    """
    column_d = pd.get_dummies(df['column'])
    column_count = len(pd.get_dummies(df['column']).columns)
    column_d.columns = [f'column{i}' for i in range(column_count)]
    column_next_d = pd.get_dummies(df['column_next'])
    column_next_d.columns = [f'column_next{i}' for i in range(column_count)]
    df = df.drop(['column', 'column_next'], axis=1)
    df = pd.merge(df, column_d, left_index=True, right_index=True)
    df = pd.merge(df, column_next_d, left_index=True, right_index=True)
    return df

def prepare_xy(df: pd.DataFrame) -> (np.ndarray, np.ndarray):
    """ Prepare the data for the model """
    df['error'] = np.abs(df['error'])
    df = df.groupby(
        ['offset', 'column', 'column_next', 'diff']
    ).median().reset_index()
    df_ = make_dummies(df.drop('r_id', axis=1))
    y = df_['error']
    X = df_.drop(['error'], axis=1)
    return X, y

#%%
map_dirs = [d for d in OSU_REPLAYS_DIR.iterdir() if d.is_dir()]
#%%
df = pd.concat([get_comparison(d) for d in (map_dirs[:5])])

#%%
# col_names = df.columns
# scaler = StandardScaler()
# scaler.fit(df,)
# df = pd.DataFrame(scaler.transform(df), columns=col_names)
X, y = prepare_xy(df)
#%%
skf = KFold(n_splits=3, shuffle=True, random_state=0)

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    rfg = RandomForestRegressor(
        verbose=2, n_estimators=100
    )
    rfg.fit(X_train, y_train)
    y_pred = rfg.predict(X_test)
    print(rfg.score(X_test, y_test))
    break

#%%
plt.hist(y_test - y_pred, bins=100)
plt.show()

