#!/bin/bash

python -m svness \
  --files_path "${FILES_DIR}" \
  --db_name "${DB_NAME}" \
  --db_username "${DB_USERNAME}" \
  --db_password "${DB_PASSWORD}" \
  --db_host "${DB_HOST}" \
  --db_port "${DB_PORT}" ||
  exit 1
