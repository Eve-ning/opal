SET @sr_min := __SR_MIN__;
SET @sr_max := __SR_MAX__;
SET @acc_min := __ACC_MIN__;
SET @acc_max := __ACC_MAX__;
SET @min_scores_per_mid := __MIN_SCORES_PER_MID__;
SET @min_scores_per_uid := __MIN_SCORES_PER_UID__;

DELIMITER //

DROP PROCEDURE IF EXISTS `dropColumnIfExists`//
CREATE PROCEDURE `dropColumnIfExists`(IN tableName VARCHAR(255), IN columnName VARCHAR(255))
BEGIN
  SELECT COUNT(*) INTO @isExists
  FROM information_schema.columns
  WHERE table_name = tableName
  AND column_name = columnName

  AND TABLE_SCHEMA = DATABASE();

  IF @isExists > 0 THEN
    SET @dsql = CONCAT('ALTER TABLE `', tableName, '` DROP `', columnName, '`');
    PREPARE stmt FROM @dsql;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
  END IF;
END //

DELIMITER ;

CALL dropColumnIfExists('osu_scores_mania_high', 'accuracy_320');
CALL dropColumnIfExists('osu_scores_mania_high', 'speed');

ALTER TABLE osu_scores_mania_high
    ADD COLUMN accuracy_320 FLOAT AS (
            (countgeki + count300 * 300 / 320 + countkatu * 200 / 320 + count100 * 100 / 320 + count50 * 50 / 320) /
            (countgeki + count300 + countkatu + count100 + count50 + countmiss)),
    ADD COLUMN speed        SMALLINT AS (CASE
                                             WHEN (`enabled_mods` >> 8 & 1) THEN -1
                                             WHEN (`enabled_mods` >> 6 & 1) THEN 1
                                             ELSE 0 END);


DROP TABLE IF EXISTS opal_beatmaps;
CREATE TABLE
    opal_beatmaps (SELECT beatmap_id mid
                   FROM osu_beatmaps b
                   WHERE (b.playmode = 3)
                     AND (b.diff_size IN (4, 7))
                     AND (b.difficultyrating BETWEEN @sr_min AND @sr_max));


DROP TABLE IF EXISTS opal_scores;
CREATE TABLE opal_scores
    (SELECT score_id     sid,
            beatmap_id   mid,
            speed        speed,
            accuracy_320 accuracy,
            user_id      uid,
            YEAR(date)   year
     FROM osu_scores_mania_high
     WHERE accuracy_320 BETWEEN @acc_min AND @acc_max);

DROP TABLE IF EXISTS opal_beatmap_scores;
CREATE TABLE opal_beatmap_scores
(
    sid      INT   NOT NULL,
    mid      INT   NOT NULL,
    speed    INT   NOT NULL,
    uid      INT   NOT NULL,
    accuracy FLOAT NOT NULL,
    year     INT   NOT NULL,
    UNIQUE INDEX idx_sid (sid),
    INDEX mid_speed_idx (mid, speed),
    INDEX uid_year_idx (uid, year)
)
SELECT s.sid      sid,
       b.mid      mid,
       s.speed    speed,
       s.uid      uid,
       s.accuracy accuracy,
       s.year     year
FROM opal_beatmaps b
         JOIN opal_scores s USING (mid);

DROP TABLE IF EXISTS opal_active_mid;
CREATE TABLE opal_active_mid
(
    mid   INT,
    speed INT,
    INDEX mid_speed_idx (mid, speed)
)
SELECT mid, speed
FROM opal_beatmap_scores
GROUP BY mid, speed
HAVING COUNT(0) > @min_scores_per_mid;

DROP TABLE IF EXISTS opal_active_uid;
CREATE TABLE opal_active_uid
(
    uid  INT,
    year INT,
    INDEX idx_mid_speed (uid, year)
)
SELECT uid, year
FROM opal_beatmap_scores
GROUP BY uid, year
HAVING COUNT(0) > @min_scores_per_uid;
