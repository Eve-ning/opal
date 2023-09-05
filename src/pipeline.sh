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

. ./utils.sh

# Preprocesses the Dataset.
# Sets the DATASET_NAME variable in the pipeline run cache.
preprocess() {
  cd preprocess || exit 1
  ./run.sh ../.env || exit 1
  cd .. || exit 1
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

preprocess || exit 1
#train || exit 1
#set -a
#source "$PIPELINE_RUN_CACHE"
#set +a
#evaluate || exit 1
#publish || exit 1
