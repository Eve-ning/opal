#!/bin/bash
mysql -h osu.mysql -P 3307 -u root -pp@ssw0rd1 -D osu <./3_preprocess.sql
