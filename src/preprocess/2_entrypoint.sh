#!/bin/bash
# Ensure that one argument is passed
if [ "$#" -ne 1 ]; then
  echo "$0 missing PIPELINE_RUN_CACHE argument"
  exit 1
fi

PIPELINE_RUN_CACHE="$1"

# Sources the run cache
. $PIPELINE_RUN_CACHE


python -m 2_svness \
--files_path "${FILES_DIR}" \
--db_name "${DB_NAME}" \
--db_username "${DB_USERNAME}" \
--db_password "${DB_PASSWORD}" \
--db_host "${DB_HOST}" \
--db_port "${DB_PORT}" \
|| exit 1
