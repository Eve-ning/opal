#!/usr/bin/env bash

# This script runs the entire pipeline, from preprocessing to publishing.
#
# Usage: ./pipeline.sh [PIPELINE_RUN_ID]
#
# The PIPELINE_RUN_ID is a unique identifier for this pipeline run.
# If not specified, it will be set to the current unix timestamp.

# Change directory to current script directory
cd "$(dirname "$(realpath "$0")")" || exit 1

# Create unique pipeline run id
PIPELINE_RUN_CACHE=.pipeline_cache/${1:-$(date +%s)}.env
mkdir -p .pipeline_cache
mkdir -p datasets
[ -f "$PIPELINE_RUN_CACHE" ] &&
  echo "Pipeline run cache ${PIPELINE_RUN_CACHE} already exists" && exit 1

# Set default values for variables
export DB_URL=https://github.com/Eve-ning/opal/raw/pipeline-automation/rsc/sample.tar.bz2
export FILES_URL=https://github.com/Eve-ning/opal/raw/pipeline-automation/rsc/sample_files.tar.bz2
cat <<EOF >>"$PIPELINE_RUN_CACHE"
PIPELINE_RUN_CACHE="$PIPELINE_RUN_CACHE"
DB_URL="$DB_URL"
FILES_URL="$FILES_URL"
FILES_DIR="/var/lib/osu/osu.files/$(basename "$FILES_URL" .tar.bz2)/"
MODEL_NAME="2023.8.4b"
DATASET_NAME="$(basename "$DB_URL" .tar.bz2)_$(date +"%Y%m%d%H%M%S").csv"
DB_NAME="osu"
DB_USERNAME="root"
DB_PASSWORD="p@ssw0rd1"
DB_HOST="osu.mysql"
DB_PORT="3307"
SR_MIN="2"
SR_MAX="15"
ACC_MIN="0.85"
ACC_MAX="1.0"
MIN_SCORES_PER_MID="0"
MIN_SCORES_PER_UID="0"
MAX_SVNESS="0.05"
EOF

# Source and Export variables
set -a
source "$PIPELINE_RUN_CACHE"
set +a

echo "Preprocessing"
envsubst < preprocess/docker-compose.yml > preprocess/docker-compose.yml.tmp
envsubst < preprocess/osu-data-docker/docker-compose.yml > preprocess/osu-data-docker/docker-compose.yml.tmp
rm preprocess/docker-compose.yml
mv preprocess/docker-compose.yml.tmp preprocess/docker-compose.yml
rm preprocess/osu-data-docker/docker-compose.yml
mv preprocess/osu-data-docker/docker-compose.yml.tmp preprocess/osu-data-docker/docker-compose.yml

docker compose \
  --profile files \
  -f preprocess/docker-compose.yml \
  up --build -d >output.log 2>&1 &

# Wait until the dataset in ./datasets/$DATASET_NAME is created
while [ ! -f "./datasets/$DATASET_NAME" ]; do
  echo "Waiting for dataset to be created... (Showing most recent log)"
  echo "$PIPELINE_RUN_CACHE"
  tail -n 3 output.log
  sleep 10
done
exit 1
docker compose \
  --profile files \
  -f preprocess/docker-compose.yml \
  --env-file preprocess/osu-data-docker/.env \
  --env-file preprocess/.env \
  stop || exit 1

source "$PIPELINE_RUN_CACHE"
[ -z "$DATASET_NAME" ] && echo "DATASET_NAME not returned by preprocess" && exit 1
export DATASET_NAME

echo "Training Model"
docker compose \
  -f train/docker-compose.yml \
  up --build || exit 1

source "$PIPELINE_RUN_CACHE"
[ -z "$MODEL_PATH" ] && echo "MODEL_PATH not returned by train" && exit 1
export MODEL_PATH

echo "Evaluating Model"
docker compose \
  -f evaluate/docker-compose.yml \
  up --build || exit 1

echo "Publishing Model"
docker compose \
  -f build/docker-compose.yml \
  up --build || exit 1
