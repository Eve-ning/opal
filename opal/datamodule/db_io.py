import os

import pandas as pd
from sqlalchemy import create_engine, text
from tqdm import tqdm

mysql_pw = os.environ['mysql_pw']


class DB:
    mysql_engine = create_engine(f'mysql+mysqlconnector://osu:{mysql_pw}@localhost:3306/osu')

    def get_df_score(
            self,
            min_active_map: int = 50,
            min_active_user: int = 50,
            accuracy_bounds: tuple[float, float] = (0.85, 1.0),
            keys: tuple[int] = (4, 7),
    ):
        """ Get Data Frame for Relevant Scores

        Args:
            min_active_map: Minimum number of scores for a beatmap to be included
            min_active_user: Minimum number of scores for a user to be included
            keys: Tuple of keys to include
            accuracy_bounds: The lower and upper bound of accuracies to include

        Returns:
            A DataFrame of:
            - score_id Index: Unique Score IDs, as per osu!'s db
            - uid: User ID
            - year: Year of Play
            - mid: Beatmap (Map) ID
            - speed: Speed of map {-1, 0, 1} to indicate {HT, NT, DT}
            - accuracy: Accuracy of play
        """
        pbar = tqdm(total=6, desc="Querying from database...", unit="Step")
        con = self.mysql_engine.connect()
        con.execute(
            text("DROP TEMPORARY TABLE IF EXISTS beatmaps, beatmap_scores, scores, active_uid, active_mid")
        )

        pbar.update()
        con.execute(text(f"""
            CREATE TEMPORARY TABLE
                beatmaps (SELECT beatmap_id mid
                          FROM osu_beatmaps b
                          WHERE (b.playmode = 3)
                            AND (b.diff_size IN {keys}));
        """))

        pbar.update()
        con.execute(text(f"""
            CREATE TEMPORARY TABLE
                scores (SELECT s.score_id     sid,
                               s.beatmap_id   mid,
                               s.speed        speed,
                               s.accuracy_320 accuracy,
                               s.user_id      uid,
                               YEAR(s.date)   year
                        FROM osu_scores_mania_high_stats_view s
                        WHERE (s.accuracy_320 BETWEEN {accuracy_bounds[0]} AND {accuracy_bounds[1]}));
        """))

        pbar.update()
        con.execute(text("""
            CREATE TEMPORARY TABLE beatmap_scores (
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
              FROM beatmaps b JOIN scores s USING (mid);
        """))

        pbar.update()
        con.execute(text(f"""
            CREATE TEMPORARY TABLE active_mid (
                mid INT,
                speed INT,
                INDEX idx_mid_speed (mid, speed)
            ) SELECT mid, speed
              FROM beatmap_scores
              GROUP BY mid, speed
              HAVING COUNT(0) > {min_active_map};
        """))

        pbar.update()
        con.execute(text(f"""
            CREATE TEMPORARY TABLE active_uid (
                uid INT,
                year INT,
                INDEX idx_mid_speed (uid, year)
            ) SELECT uid, year
              FROM beatmap_scores
              GROUP BY uid, year
              HAVING COUNT(0) > {min_active_user};
        """))

        pbar.update()
        df = pd.read_sql(
            "SELECT * FROM beatmap_scores NATURAL JOIN active_mid NATURAL JOIN active_uid;",
            con=con
        ).set_index('sid')

        # Clean up temp tables
        con.execute(
            text("DROP TEMPORARY TABLE IF EXISTS beatmaps, beatmap_scores, scores, active_uid, active_mid")
        )
        return df
