#!/bin/bash

# Ensure all variables are set
: "${DATASET_NAME:?DATASET_NAME not set}"
: "${MODEL_NAME:?MODEL_NAME not set}"
: "${PIPELINE_RUN_CACHE:?PIPELINE_RUN_CACHE not set}"

# TODO: Figure out a way for opal.train to set env var in shell instead of within python
#       An idea is to specify the model version number `version_XXXX` then we can grep for it.
# This internally sets the MODEL_PATH env var in the pipeline run cache
python -m opal.train \
  --dataset_name "$DATASET_NAME" \
  --model_name "$MODEL_NAME" \
  --pipeline_run_cache "$PIPELINE_RUN_CACHE" ||
  exit 1
