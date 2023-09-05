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

. ./utils

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
  set -a
  source .env
  set +a
}

make_pipeline_cache "$1" || exit 1
load_env || exit 1
preprocess || exit 1
train || exit 1
set -a
source "$PIPELINE_RUN_CACHE"
set +a
evaluate || exit 1
publish || exit 1
