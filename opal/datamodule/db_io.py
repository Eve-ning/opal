import os
from pathlib import Path

import numpy as np
import pandas as pd
from reamber.algorithms.analysis import scroll_speed
from reamber.osu import OsuMap
from sqlalchemy import create_engine, text
from tqdm import tqdm

mysql_pw = os.environ['mysql_pw']


class DB:
    mysql_engine = create_engine(f'mysql+mysqlconnector://osu:{mysql_pw}@localhost:3306/osu')

    def __init__(
            self,
            osu_files_path: Path,
            min_active_map: int = 50,
            min_active_user: int = 50,
            accuracy_bounds: tuple[float, float] = (0.85, 1.0),
            keys: tuple[int] = (4, 7),
            visual_complexity_limit: float = 0.05,
            regen_tables: bool = False,
    ):
        """ Initializes a database instance to interact with MySQL

        Args:
            min_active_map: Minimum number of scores for a beatmap to be included
            min_active_user: Minimum number of scores for a user to be included
            keys: Tuple of keys to include
            accuracy_bounds: The lower and upper bound of accuracies to include
            visual_complexity_limit: The upper limit of visual complexity. Ranges [0, 1)
            regen_tables: Whether to regenerate the auxiliary opal MySQL tables.
                Will automatically be set to true if they don't exist.
        """

        self.visual_complexity_limit = visual_complexity_limit

        # Check if last table is generated
        if self.get_table("opal_beatmaps_visual_complexity") is None:
            regen_tables = True

        if regen_tables:
            con = self.mysql_engine.connect()
            pbar = tqdm(total=6, desc="Initializating Temporary Tables...")

            pbar.update()
            con.execute(text("DROP TABLE IF EXISTS opal_beatmaps"))
            con.execute(text(f"""
                CREATE TABLE
                    opal_beatmaps (SELECT beatmap_id mid
                                   FROM osu_beatmaps b
                                   WHERE (b.playmode = 3)
                                     AND (b.diff_size IN {keys}));
            """))

            pbar.update()
            con.execute(text("DROP TABLE IF EXISTS opal_scores"))
            con.execute(text(f"""
                CREATE TABLE
                    opal_scores (SELECT s.score_id     sid,
                                        s.beatmap_id   mid,
                                        s.speed        speed,
                                        s.accuracy_320 accuracy,
                                        s.user_id      uid,
                                        YEAR(s.date)   year
                            FROM osu_scores_mania_high_stats_view s
                            WHERE (s.accuracy_320 BETWEEN {accuracy_bounds[0]} AND {accuracy_bounds[1]}));
            """))

            pbar.update()
            con.execute(text("DROP TABLE IF EXISTS opal_beatmap_scores"))
            con.execute(text("""
                CREATE TABLE opal_beatmap_scores (
                    sid INT NOT NULL,
                    mid INT NOT NULL,
                    speed INT NOT NULL,
                    uid INT NOT NULL,
                    accuracy FLOAT NOT NULL,
                    year INT NOT NULL,
                    UNIQUE INDEX idx_sid (sid),
                    INDEX idx_mid_speed (mid, speed),
                    INDEX idx_uid_year (uid, year)
                ) SELECT s.sid      sid,
                         b.mid      mid,
                         s.speed    speed,
                         s.uid      uid,
                         s.accuracy accuracy,
                         s.year     year
                  FROM opal_beatmaps b JOIN opal_scores s USING (mid);
            """))

            pbar.update()
            con.execute(text("DROP TABLE IF EXISTS opal_active_mid"))
            con.execute(text(f"""
                CREATE TABLE opal_active_mid (
                    mid INT,
                    speed INT,
                    INDEX idx_mid_speed (mid, speed)
                ) SELECT mid, speed
                  FROM opal_beatmap_scores
                  GROUP BY mid, speed
                  HAVING COUNT(0) > {min_active_map};
            """))

            pbar.update()
            con.execute(text("DROP TABLE IF EXISTS opal_active_uid"))
            con.execute(text(f"""
                CREATE TABLE opal_active_uid (
                    uid INT,
                    year INT,
                    INDEX idx_mid_speed (uid, year)
                ) SELECT uid, year
                  FROM opal_beatmap_scores
                  GROUP BY uid, year
                  HAVING COUNT(0) > {min_active_user};
            """))

            pbar.update()
            pbar.close()
            con.execute(text("DROP TABLE IF EXISTS opal_beatmaps_visual_complexity"))
            self.create_osu_beatmaps_visual_complexity(osu_files_path)
            con.close()

    def get_df_score(self) -> pd.DataFrame:
        """ Get Data Frame for Relevant Scores

        Returns:
            A DataFrame of:
            - score_id Index: Unique Score IDs, as per osu!'s db
            - uid: User ID
            - year: Year of Play
            - mid: Beatmap (Map) ID
            - speed: Speed of map {-1, 0, 1} to indicate {HT, NT, DT}
            - accuracy: Accuracy of play
            - vc_ix: Visual Complexity
        """
        con = self.mysql_engine.connect()
        df_beatmap_scores = pd.read_sql_table("opal_beatmap_scores", con=con)
        df_active_mid = pd.read_sql_table("opal_active_mid", con=con)
        df_active_uid = pd.read_sql_table("opal_active_uid", con=con)
        df_vc = pd.read_sql_table("opal_beatmaps_visual_complexity", con=con)
        df_vc = df_vc.loc[df_vc['visual_complexity'] < self.visual_complexity_limit]

        return (
            df_beatmap_scores
            .merge(df_active_mid, on=['mid', 'speed'])
            .merge(df_active_uid, on=['uid', 'year'])
            .merge(df_vc, on=['mid'])
        )

    def create_osu_beatmaps_visual_complexity(
            self,
            osu_files_path: Path,
    ) -> pd.DataFrame:
        """ Creates the Visual Complexity SQL Table within MySQL

        Args:
            osu_files_path: Path to the osu files directory.

        Returns:
            The beatmap visual complexity table.
        """

        # Pull the beatmap ids we're interested in
        con = self.mysql_engine.connect()
        mids = pd.read_sql_table("opal_active_mid", con=con)['mid'].unique()

        # Create the DF we'll populate with vc_ix
        df = pd.DataFrame(dict(mid=mids))

        def visual_complexity(x):
            # Visual Complexity Equation
            # Below 1.0, we use a simple x^2
            # Above 1.0, we do a sigmoid easing from 1 to 3
            # Otherwise, we set to 1.0
            return np.piecewise(
                x,
                [
                    (0 <= x) * (x < 1),
                    (1 <= x) * (x < 3)
                ],
                [
                    lambda x: (x - 1) ** 2,  # x^2 0 to 1
                    lambda x: np.sin((x - 2) * np.pi * 0.5) / 2 + 0.5,  # Sigmoid Easing 1 to 3
                    1  # Otherwise
                ]
            )

        def osu_visual_complexity(osu):
            speed = scroll_speed(osu)
            offsets = speed.index

            return np.sum(
                # Evaluate the integral visual complexity w.r.t. time
                (visual_complexity(speed.to_numpy()[:-1]) * np.diff(offsets)) /
                # Take the proportional visual complexity
                (offsets.max() - offsets.min())
            )

        for mid in tqdm(mids, desc="Evaluating Visual Complexity of Maps..."):
            # Get our osu map
            osu_path = osu_files_path / f"{mid}.osu"
            osu = OsuMap.read_file(osu_path)

            # Get visual_complexity
            vc = osu_visual_complexity(osu)

            # Set the Visual Complexity to corresponding map
            df.loc[df['mid'] == mid, 'visual_complexity'] = vc

        df = df.set_index('mid')

        # Send to sql
        df.to_sql(
            name="opal_beatmaps_visual_complexity",
            con=self.mysql_engine,
            if_exists='replace',
        )
        con.close()
        return df

    def get_table(self, table_name):
        con = self.mysql_engine.connect()
        try:
            sql_table = pd.read_sql_table(table_name, con=con)
        except ValueError:
            sql_table = None

        con.close()

        return sql_table
