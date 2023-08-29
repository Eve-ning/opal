#!/usr/bin/env bash

# This script runs the entire pipeline, from preprocessing to publishing.
#
# Usage: ./pipeline.sh [PIPELINE_RUN_ID]
#
# The PIPELINE_RUN_ID is a unique identifier for this pipeline run.
# If not specified, it will be set to the current unix timestamp.

# Dev Info
# On the Docker Compose Substitution:
# While working locally, docker compose environmental substitution is extremely janky on GH Actions.
# This is especially true using the `include` directive in our preprocess block.
# To play it safe, we use a custom function `envdotsub` to substitute the environment variables.

# Change directory to current script directory
cd "$(dirname "$(realpath "$0")")" || exit 1

# Substitute environment variables in a file
envdotsub() {
  filename=$(basename "$1")
  dir=$(dirname "$1")
  dotfile=".$filename"
  dotfilepath="$dir/$dotfile"
  envsubst <"$1" >"$dotfilepath"
}

# Preprocesses the Dataset.
# Sets the DATASET_NAME variable in the pipeline run cache.
preprocess() {
  envdotsub preprocess/docker-compose.yml
  sed -i 's|osu-data-docker/docker-compose.yml|osu-data-docker/.docker-compose.yml|g' \
   preprocess/.docker-compose.yml || exit 1
  envdotsub preprocess/osu-data-docker/docker-compose.yml

  # Without -d, this script will hang until the docker compose process is killed
  # To also include compose stop, we'll peek at the dataset file and wait for it to be created
  docker compose \
    --profile files \
    -f preprocess/.docker-compose.yml \
    up --build -d >preprocess.log 2>&1 &

  while [ ! -f "./datasets/$DATASET_NAME" ]; do
    echo "Waiting for dataset to be created... (Showing most recent log)"

    if grep -q "exit 1" preprocess.log; then
      echo "Preprocess exited with errors. See preprocess.log."
      exit 1
    fi

    tail -n 3 preprocess.log
    sleep 10
  done

  docker compose \
    --profile files \
    -f preprocess/.docker-compose.yml \
    stop || exit 1

  source "$PIPELINE_RUN_CACHE"
  if [ -z "$DATASET_NAME" ]; then
    echo "DATASET_NAME not returned by preprocess"
    exit 1
  fi
}

# Trains the Model.
# Sets the MODEL_PATH variable in the pipeline run cache.
train() {
  envdotsub train/docker-compose.yml
  docker compose \
    -f train/.docker-compose.yml \
    up --build || exit 1

  source "$PIPELINE_RUN_CACHE"
  if [ -z "$MODEL_PATH" ]; then
    echo "MODEL_PATH not returned by train"
    exit 1
  fi
}

# Evaluates the Model.
evaluate() {
  echo "Evaluating Model"
  envdotsub evaluate/docker-compose.yml
  docker compose \
    -f evaluate/.docker-compose.yml \
    up --build || exit 1
}

# Publishes the Model via PyPI.
publish() {
  echo "Publishing Model"
  envdotsub build/docker-compose.yml
  docker compose \
    -f build/.docker-compose.yml \
    up --build || exit 1
}

make_pipeline_cache() {
  # Create unique pipeline run id
  PIPELINE_RUN_CACHE=.pipeline_cache/${1:-$(date +%s)}.env
  mkdir -p .pipeline_cache
  mkdir -p datasets
  if [ -f "$PIPELINE_RUN_CACHE" ]; then
    echo "Pipeline run cache ${PIPELINE_RUN_CACHE} already exists"
    exit 1
  fi
}

load_env() {
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
  source preprocess/osu-data-docker/.env
  set +a
}

make_pipeline_cache "$1" || exit 1
load_env || exit 1
preprocess || exit 1
train || exit 1
evaluate || exit 1
publish || exit 1
