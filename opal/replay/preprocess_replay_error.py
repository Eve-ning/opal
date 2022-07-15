from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from reamber.algorithms.osu.OsuReplayError import osu_replay_error
from reamber.algorithms.pattern import Pattern
from reamber.algorithms.pattern.combos import PtnCombo
from reamber.osu.OsuMap import OsuMap


def preprocess_replay_error(map_dir: Path) -> pd.DataFrame:
    return PreprocessReplayError(map_dir).df()


@dataclass
class PreprocessReplayError:
    map_dir: Path

    @staticmethod
    def get_errors(osu: OsuMap, rep_paths: List[Path]) -> pd.DataFrame:
        """ Gets the errors of the replays.

        Args:
            rep_paths: Paths of the replays
            osu: Path of the osu! map

        """
        rep_paths = [p for p in rep_paths if Path(p).stat().st_size > 1000]
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

    @staticmethod
    def get_pattern(osu: OsuMap) -> pd.DataFrame:
        """ Gets the pattern of the map. """
        groups = Pattern.from_note_lists(
            [osu.hits, osu.holds], False
        ).group()
        combos = np.concatenate(PtnCombo(groups).combinations(size=2), axis=0)
        return pd.DataFrame({
            "offset": combos['offset'][:, 0],
            "column": combos['column'][:, 0],
            "column_next": combos['column'][:, 1],
            "diff": np.diff(combos['offset'], axis=1).squeeze(),
        }).astype(int)

    @staticmethod
    def make_dummies(df: pd.DataFrame) -> pd.DataFrame:
        """ Convert column variables to dummies

        Args:
            df: DataFrame with column variables
        """
        column_d = pd.get_dummies(df['column'])
        column_count = len(pd.get_dummies(df['column']).columns)
        column_d.columns = [f'column{i}' for i in range(column_count)]
        column_next_d = pd.get_dummies(df['column_next'])
        column_next_d.columns = [f'column_next{i}' for i in
                                 range(column_count)]
        df = df.drop(['column', 'column_next'], axis=1)
        df = pd.merge(df, column_d, left_index=True, right_index=True)
        df = pd.merge(df, column_next_d, left_index=True, right_index=True)
        return df

    def df(self) -> pd.DataFrame:
        """ Prepare the data for the model """
        map_path = self.map_dir / (self.map_dir.name + ".osu")
        osu = OsuMap.read_file(map_path.as_posix())
        rep_dir = self.map_dir / "rep"
        rep_paths = [p.as_posix() for p in rep_dir.iterdir() if p.is_file()]
        df = pd.merge(self.get_pattern(osu), self.get_errors(osu, rep_paths),
                      how='left', on=['column', 'offset'])

        df['error'] = np.abs(df['error'])
        df = df.groupby(['offset', 'column', 'column_next', 'diff']) \
            .median().reset_index()
        return self.make_dummies(df.drop('r_id', axis=1))
