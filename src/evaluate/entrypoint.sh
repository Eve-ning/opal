#!/usr/bin/env bash

# Ensure all variables are set
: "${MODEL_PATH:?MODEL_PATH not set}"
: "${DATASET_NAME:?DATASET_NAME not set}"

python -m opal.evaluate \
  --model_path "$MODEL_PATH" \
  --dataset_name "$DATASET_NAME" ||
  exit 1
