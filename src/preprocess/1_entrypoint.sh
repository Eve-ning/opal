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

# Ping mysql until it's ready with 3 retries of 5s interval
for _ in {1..3}; do
  if mysqladmin ping -h osu.mysql -P 3307 -u root -pp@ssw0rd1; then
    break
  fi
  echo "Waiting for osu.mysql to be ready..."
  sleep 5
done

echo "$SQL_SCRIPT" | mysql -h osu.mysql -P 3307 -u root -pp@ssw0rd1 -D osu
