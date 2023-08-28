#!/bin/bash

SQL_SCRIPT=$(cat ./1_preprocess.sql)

: "${SR_MIN:=2}"
: "${SR_MAX:=15}"
: "${ACC_MIN:=0.85}"
: "${ACC_MAX:=1.0}"
: "${MIN_SCORES_PER_MID:=50}"
: "${MIN_SCORES_PER_UID:=50}"

SQL_SCRIPT=${SQL_SCRIPT//__SR_MIN__/"$SR_MIN"}
SQL_SCRIPT=${SQL_SCRIPT//__SR_MAX__/"$SR_MAX"}
SQL_SCRIPT=${SQL_SCRIPT//__ACC_MIN__/"$ACC_MIN"}
SQL_SCRIPT=${SQL_SCRIPT//__ACC_MAX__/"$ACC_MAX"}
SQL_SCRIPT=${SQL_SCRIPT//__MIN_SCORES_PER_MID__/"$MIN_SCORES_PER_MID"}
SQL_SCRIPT=${SQL_SCRIPT//__MIN_SCORES_PER_UID__/"$MIN_SCORES_PER_UID"}

# Ping mysql until it's ready with 5s interval
until mysqladmin ping -h osu.mysql -P 3307 -u root -pp@ssw0rd1; do
  echo "Waiting for osu.mysql to be ready..."
  sleep 5
done

# Wait until tables are ready
until mysql -h osu.mysql -P 3307 -u root -pp@ssw0rd1 -D osu \
  -e 'SELECT * FROM osu_scores_mania_high LIMIT 1;' >>/dev/null 2>&1; do
  mysql -h osu.mysql -P 3307 -u root -pp@ssw0rd1 -D osu -e "SHOW TABLES;"
  echo "Waiting for osu_scores_mania_high to be ready..."
  sleep 5
done
until mysql -h osu.mysql -P 3307 -u root -pp@ssw0rd1 -D osu \
  -e 'SELECT * FROM osu_beatmaps LIMIT 1;' >>/dev/null 2>&1; do
  mysql -h osu.mysql -P 3307 -u root -pp@ssw0rd1 -D osu -e "SHOW TABLES;"
  echo "Waiting for osu_beatmaps to be ready..."
  sleep 5
done

echo "$SQL_SCRIPT" | mysql -h osu.mysql -P 3307 -u root -pp@ssw0rd1 -D osu
