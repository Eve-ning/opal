from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
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
            sr_bounds: tuple[float, float] = (2.5, 15),
            keys: tuple[int] = (4, 7),
            visual_complexity_limit: float = 0.05,
    ):
        """ Initializes a database instance to interact with MySQL

        Args:
            min_active_map: Minimum number of scores for a beatmap to be included
            min_active_user: Minimum number of scores for a user to be included
            keys: Tuple of keys to include
            accuracy_bounds: The lower and upper bound of accuracies to include
            sr_bounds: The lower and upper bound of star ratings to include
            visual_complexity_limit: The upper limit of visual complexity. Ranges [0, 1)
        """

        self.visual_complexity_limit = visual_complexity_limit

        con = self.mysql_engine.connect()
        pbar = tqdm(total=0, desc="Initializating Temporary Tables...")

        pbar.set_description("Checking opal_beatmaps")
        if self.get_table("opal_beatmaps", 1) is None:
            con.execute(text("DROP TABLE IF EXISTS opal_beatmaps"))
            con.execute(text(f"""
                CREATE TABLE
                    opal_beatmaps (
                    SELECT beatmap_id mid
                    FROM osu_beatmaps b
                    WHERE (b.playmode = 3)
                    AND (b.diff_size IN {keys})
                    AND (b.difficultyrating BETWEEN {sr_bounds[0]} AND {sr_bounds[1]})
                    );
            """))

        pbar.set_description("Checking opal_scores")
        if self.get_table("opal_scores", 1) is None:
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

        pbar.set_description("Checking opal_beatmap_scores")
        if self.get_table("opal_beatmap_scores", 1) is None:
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

        pbar.set_description("Checking opal_active_mid")
        if self.get_table("opal_active_mid", 1) is None:
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

        pbar.set_description("Checking opal_active_uid")
        if self.get_table("opal_active_uid", 1) is None:
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

        pbar.set_description("Checking opal_beatmaps_visual_complexity")
        if self.get_table("opal_beatmaps_visual_complexity", 1) is None:
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
        df = pd.read_sql_table("opal_active_scores", con=con)
        df_vc = pd.read_sql_table("opal_beatmaps_visual_complexity", con=con)
        df_vc = df_vc.loc[df_vc['visual_complexity'] < self.visual_complexity_limit]

        return df.merge(df_vc, on=['mid'])

    def get_table(self, table_name, limit: int | None = None):
        con = self.mysql_engine.connect()
        try:
            result = pd.read_sql_query(f"SELECT * FROM {table_name} {f'LIMIT {limit}' if limit else ''};",
                                       con=con)
            con.close()
        except:
            con.rollback()
            result = None
        return result
