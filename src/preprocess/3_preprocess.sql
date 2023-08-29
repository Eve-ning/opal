SET @max_svness := __MAX_SVNESS__;

DROP TABLE IF EXISTS opal_active_mid_svness_low;
CREATE TABLE opal_active_mid_svness_low
(
    mid    INT,
    speed  FLOAT,
    INDEX mid (mid, speed)
)
SELECT oams.mid AS mid,
       speed    AS speed
FROM opal_active_mid_svness oams
         JOIN osu.opal_active_mid oam ON oams.mid = oam.mid
WHERE oams.svness < @max_svness;

DROP TABLE IF EXISTS opal_active_scores;
CREATE TABLE opal_active_scores (SELECT DISTINCT s.sid,
                                                 s.mid,
                                                 s.speed,
                                                 s.uid,
                                                 s.accuracy,
                                                 s.year
                                 FROM opal_beatmap_scores s
                                          JOIN opal_active_mid_svness_low oams
                                              ON s.mid = oams.mid AND s.speed = oams.speed
                                          JOIN opal_active_uid oau ON s.uid = oau.uid);
