#!/usr/bin/env bash

# Ensure all variables are set
: "${DATASET_NAME:?DATASET_NAME not set}"

mysql -h osu.mysql -P 3307 -u root -pp@ssw0rd1 -D osu < \
./4_export.sql | \
sed 's/\t/,/g' >../datasets/"${DATASET_NAME}"

exit 0