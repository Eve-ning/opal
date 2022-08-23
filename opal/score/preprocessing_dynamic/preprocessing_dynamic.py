from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


def report(f):
    def report_wrapper(self=None):
        print(f"{f.__name__} {len(self.df)} -> ", end="")
        f(self)
        print(len(self.df))

    return report_wrapper


@dataclass
class PreprocessingDynamic:
    df: pd.DataFrame
    unpopular_maps_thres: float | None = 0.2
    unpopular_plays_thres: float | None = 0.2
    score_year_filter: Tuple[int, int] | None = (2019, 2022)
    sr_min_thres: float | None = 4.5
    score_filter: Tuple[int, int] | None = (600_000, 990_000)
    acc_filter: Tuple[float, float] | None = (0.95, 1.0)
    remove_mod: bool | None = True
    sample_users: int | None = None
    acc_filter_300g_weight: int = 315

    @report
    def by_sample_users(self):
        if self.sample_users:
            uid = self.df['user_id']
            self.df = self.df[
                uid.isin(np.random.choice(uid.unique(), self.sample_users))
            ]

    @report
    def by_score_year(self):
        self.df = self.df.loc[
            (self.df['year'] >= self.score_year_filter[0]) &
            (self.df['year'] < self.score_year_filter[1])
            ]

    @report
    def by_sr(self):
        self.df = self.df.loc[
            self.df['sr'] >= self.sr_min_thres
            ]

    @report
    def by_unpopular_maps(self):
        user_count = len(self.df['user_id'].unique())
        df_g = self.df.groupby('map_id')
        self.df = df_g.filter(
            lambda x:
            len(x) >= (user_count * self.unpopular_maps_thres)
        )

    @report
    def by_unpopular_plays(self):
        beatmap_count = len(self.df['map_id'].unique())
        df_g = self.df.groupby(['user_id', 'year', 'mod'])
        self.df = df_g.filter(
            lambda x:
            len(x) >= (beatmap_count * self.unpopular_plays_thres)
        )

    @report
    def by_score_filter(self):
        self.df = self.df.loc[
            (self.df['score'] >= self.score_filter[0]) &
            (self.df['score'] < self.score_filter[1])
            ]

    @report
    def by_acc_filter(self):
        if 'accuracy' not in self.df.columns: self.calc_acc()

        self.df = self.df.loc[
            (self.df['accuracy'] >= self.acc_filter[0]) &
            (self.df['accuracy'] < self.acc_filter[1])
            ]

    @report
    def by_remove_mod(self):
        if self.remove_mod:
            self.df = self.df.loc[self.df['mod'] == 0]

    def filter(self, calc_acc: bool = True):

        if calc_acc: self.calc_acc()
        if self.sample_users is not None: self.by_sample_users()
        if self.score_year_filter is not None: self.by_score_year()
        if self.sr_min_thres is not None: self.by_sr()
        if self.unpopular_maps_thres is not None: self.by_unpopular_maps()
        if self.unpopular_plays_thres is not None: self.by_unpopular_plays()
        if self.score_filter is not None: self.by_score_filter()
        if self.acc_filter is not None: self.by_acc_filter()
        if self.remove_mod is not None: self.by_remove_mod()

        user_count = len(self.df['user_id'].unique())
        beatmap_count = len(self.df['map_id'].unique())

        print(f"Users Left: {user_count} | Beatmaps Left: {beatmap_count}")

        return self.df

    def calc_acc(self):
        count = self.df['count300g'] + \
                self.df['count300'] + \
                self.df['count200'] + \
                self.df['count100'] + \
                self.df['count50'] + \
                self.df['count0']

        self.df['accuracy'] = \
            (
                self.df['count300g'] +
                self.df['count300'] * 300 / self.acc_filter_300g_weight +
                self.df['count200'] * 200 / self.acc_filter_300g_weight +
                self.df['count100'] * 100 / self.acc_filter_300g_weight +
                self.df['count50'] * 50 / self.acc_filter_300g_weight
            ) / count
