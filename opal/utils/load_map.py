from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from reamber.algorithms.osu.OsuReplayError import osu_replay_error
from reamber.algorithms.pattern import Pattern
from reamber.base.Hold import HoldTail
from reamber.osu.OsuHit import OsuHit
from reamber.osu.OsuHold import OsuHold
from reamber.osu.OsuMap import OsuMap


def load_map(map_dir: Path) -> Tuple[pd.DataFrame, OsuMap]:
    return MapLoader(map_dir).load()


@dataclass
class MapLoader:
    map_dir: Path

    @staticmethod
    def get_errors(osu: OsuMap, rep_paths: List[Path]) -> pd.DataFrame:
        """ Gets the errors of the replays.

        Args:
            rep_paths: Paths of the replays
            osu: Path of the osu! map
        """

        # Filter bad replays
        rep_paths = [p for p in rep_paths if p.stat().st_size > 1000]

        # Yield Replay Error
        errors = osu_replay_error([r.as_posix() for r in rep_paths], osu)

        # Get map offsets regardless of type
        # k_o key offsets
        df_map_offset = pd.DataFrame.from_records(
            [(o,)
             for _, k_o in [*errors.map_offsets.hits.items(),
                          *errors.map_offsets.releases.items()]
             for o in k_o],
            columns=["offset"]
        )

        # Get replay errors as offset
        df_errors = pd.DataFrame.from_records(
            [(r_id, k, o)
             for r_id, rep_offset in enumerate(errors.errors)
             for k, k_o in [*rep_offset.hits.items(),
                          *rep_offset.releases.items()]
             for o in k_o],
            columns=["r_id", "column", "error"]
        )

        return pd.merge(
            # We combine map offsets & error
            # For n replays, we repeat the map offsets n times
            pd.concat([df_map_offset] * len(rep_paths)).reset_index(),
            df_errors,
            left_index=True,
            right_index=True
        ).drop('index', axis=1).astype(int).assign(
            # Absolute error
            error=lambda x: x.error.abs()
        ).drop(['r_id', 'column'], axis=1).groupby(
            # Get the median error
            ['offset']
        ).median().reset_index()

    @staticmethod
    def get_pattern(osu: OsuMap) -> pd.DataFrame:
        """ Gets the pattern of the map. """

        grps = Pattern.from_note_lists([osu.hits, osu.holds]).group()

        # Manually extract if columns are held
        is_held = []
        grps_hold = []
        for grp in grps:
            holds = [note.column for note in grp if note.type == OsuHold]
            hits = [note.column for note in grp if note.type == OsuHit]
            tails = [note.column for note in grp if note.type == HoldTail]

            grps_hold.append(
                [np.min(grp.offset), [*hits, *holds], deepcopy(is_held)]
            )
            is_held.extend(holds)
            is_held = [c for c in is_held if c not in tails]

        df = pd.DataFrame(grps_hold, columns=["offset", "columns", "is_held"])

        # OHE for bigram
        df_cols = pd.get_dummies(
            df['columns'].apply(pd.Series, dtype=int).stack()).groupby(
            level=0
        ).sum()
        df_cols.columns = [f'col_{c}' for c in range(len(df_cols.columns))]

        # OHE for held
        df_hold = pd.get_dummies(
            df['is_held'].apply(pd.Series, dtype=int).stack()).groupby(
            level=0
        ).sum()
        df_hold.columns = [f'is_held_{c}' for c in range(len(df_hold.columns))]

        return (
            pd.merge(
                # Horizontally Join Offset & Cols
                df['offset'], df_cols, how='left', left_index=True,
                right_index=True
            ).merge(
                # Horizontally Join Offset, Cols & Held
                df_hold, how='left', left_index=True,
                right_index=True
            ).fillna(
                0
            ).assign(
                # Assign Delta to diff
                diff=lambda x: x['offset'].diff().shift(-1)
            )[:-1]
        )

    def load(self) -> Tuple[pd.DataFrame, OsuMap]:
        """ Prepare the data for the model """
        map_path = self.map_dir / (self.map_dir.name + ".osu")
        osu = OsuMap.read_file(map_path.as_posix())
        rep_dir = self.map_dir / "rep"
        rep_paths = [p for p in rep_dir.iterdir() if p.is_file()]

        return pd.merge(
            self.get_pattern(osu),
            self.get_errors(osu, rep_paths),
            how='left',
            on='offset'
        ).drop(['offset'], axis=1), osu
