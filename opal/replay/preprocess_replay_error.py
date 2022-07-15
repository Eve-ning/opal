from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from reamber.algorithms.osu.OsuReplayError import osu_replay_error
from reamber.algorithms.pattern import Pattern
from reamber.base.Hold import HoldTail
from reamber.osu.OsuHit import OsuHit
from reamber.osu.OsuHold import OsuHold
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
        rep_paths = [p for p in rep_paths if p.stat().st_size > 1000]
        errors = osu_replay_error([r.as_posix() for r in rep_paths], osu)
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
        ).drop('index', axis=1).astype(int).assign(
            error=lambda x: x.error.abs()
        ).drop(['r_id', 'column'], axis=1) \
            .groupby(['offset']).median().reset_index()

    @staticmethod
    def get_pattern(osu: OsuMap) -> pd.DataFrame:
        """ Gets the pattern of the map. """
        groups = Pattern.from_note_lists([osu.hits, osu.holds]).group()
        is_held = []

        groups_new = []
        for group in groups:
            holds = [note.column for note in group if note.type == OsuHold]
            hits = [note.column for note in group if note.type == OsuHit]
            tails = [note.column for note in group if note.type == HoldTail]

            groups_new.append(
                [np.min(group.offset),
                 [*hits, *holds], deepcopy(is_held)]
            )
            is_held.extend(holds)
            is_held = [c for c in is_held if c not in tails]
        df = pd.DataFrame(groups_new, columns=["offset", "columns", "is_held"])
        df_cols = pd.get_dummies(
            df['columns'].apply(pd.Series, dtype=int).stack()).groupby(
            level=0
        ).sum()
        df_cols.columns = [f'col_{c}' for c in range(len(df_cols.columns))]
        df_hold = pd.get_dummies(
            df['is_held'].apply(pd.Series, dtype=int).stack()).groupby(
            level=0
        ).sum()
        df_hold.columns = [f'hold_{c}' for c in range(len(df_hold.columns))]
        return (
            pd.merge(
                df['offset'], df_cols, how='left', left_index=True,
                right_index=True
            ).merge(
                df_hold, how='left', left_index=True,
                right_index=True
            ).fillna(
                0
            ).assign(
                diff=lambda x: x['offset'].diff().shift(-1)
            )[:-1]
        )

    def df(self) -> pd.DataFrame:
        """ Prepare the data for the model """
        map_path = self.map_dir / (self.map_dir.name + ".osu")
        osu = OsuMap.read_file(map_path.as_posix())
        rep_dir = self.map_dir / "rep"
        rep_paths = [p for p in rep_dir.iterdir() if p.is_file()]

        return pd.merge(
            self.get_pattern(osu),
            self.get_errors(osu, rep_paths), how='left',
            on='offset'
        ).drop(['offset'], axis=1)
