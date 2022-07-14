from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reamber.algorithms.osu.OsuReplayError import osu_replay_error
from reamber.algorithms.pattern import Pattern
from reamber.algorithms.pattern.combos import PtnCombo
from reamber.algorithms.playField import PlayField
from reamber.algorithms.playField.parts import *
from reamber.osu.OsuMap import OsuMap

from opal.conf.conf import OSU_REPLAYS_DIR

map_dirs = [d for d in OSU_REPLAYS_DIR.iterdir() if d.is_dir()]
map_dir = map_dirs[48]

map_path = map_dir / (map_dir.name + ".osu")
osu = OsuMap.read_file(map_path)
rep_dir = map_dir / "rep"
rep_paths = [p.as_posix() for p in rep_dir.iterdir() if p.is_file()]


def get_errors(rep_paths: List[str], osu: OsuMap) -> pd.DataFrame:
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
    return pd.merge(pd.concat([df_map_offset] * rep_count).reset_index(),
                    df_errors,
                    left_index=True, right_index=True).astype(int)

def get_pattern(osu: OsuMap) -> pd.DataFrame:
    groups = Pattern.from_note_lists([osu.hits, osu.holds], False).group()
    combos = np.concatenate(PtnCombo(groups).combinations(size=2), axis=0)
    return pd.DataFrame({
        "offset": combos['offset'][:, 0],
        "column": combos['column'][:, 0],
        "column_next": combos['column'][:, 1],
        "diff": np.diff(combos['offset'], axis=1).squeeze(),
    }).astype(int)

# %%
df = pd.merge(get_pattern(osu), get_errors(rep_paths, osu),
              how='left', on=['column', 'offset'])
#%%
df_ = df[['offset', 'error']]
df_['error'] = np.abs(df_['error'])
df_ = df_.groupby(['offset']).median()

plt.figure(figsize=(20,3))
conv = 10
plt.plot(np.convolve(df_.index, np.ones([conv]) / conv, 'valid'),
         np.convolve(df_['error'], np.ones([conv]) / conv, 'valid'))
# plt.ylim([0, 100])
plt.show()
# pf = (
#     PlayField(osu) +
#     PFDrawColumnLines() +
#     PFDrawBeatLines() +
#     PFDrawNotes() +
#     PFDrawLines.from_combo(
#         combos,
#         4,
#         **PFDrawLines.Colors.RED)
# )
#
# pf.export_fold(2000).save("test.png")