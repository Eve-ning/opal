SET @max_svenss := 0.05;

DROP TABLE IF EXISTS opal_active_mid_svness_low;
CREATE TABLE opal_active_mid_svness_low
(
    mid    INT,
    svness FLOAT,
    INDEX mid (mid)
)
SELECT oams.mid AS mid,
       speed    AS speed,
       svness   AS svness
FROM opal_active_mid_svness oams
         JOIN osu.opal_active_mid oam ON oams.mid = oam.mid
WHERE oams.svness < @max_svenss;

DROP TABLE IF EXISTS opal_active_scores;
CREATE TABLE opal_active_scores (SELECT DISTINCT s.sid,
                                                 s.mid,
                                                 s.speed,
                                                 s.uid,
                                                 s.accuracy,
                                                 s.year
                                 FROM opal_beatmap_scores s
                                          JOIN opal_active_mid_svness_low oams ON s.mid = oams.mid
                                          JOIN opal_active_uid oau ON s.uid = oau.uid);
