#!/bin/bash
EXPORT_NAME="$1"

mysql -h osu.mysql -P 3307 -u root -pp@ssw0rd1 -D osu < \
./4_export.sql | \
sed 's/\t/,/g' >../datasets/"${EXPORT_NAME}"
