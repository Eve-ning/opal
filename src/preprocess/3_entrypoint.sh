#!/usr/bin/env bash

SQL_SCRIPT=$(cat ./3_preprocess.sql)

: "${MAX_SVNESS:=0.05}"

SQL_SCRIPT=${SQL_SCRIPT//__MAX_SVNESS__/"$MAX_SVNESS"}

echo "$SQL_SCRIPT" | mysql -h osu.mysql -P 3307 -u root -pp@ssw0rd1 -D osu
