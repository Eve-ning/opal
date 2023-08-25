#!/bin/bash
# Ensure that one argument is passed
if [ "$#" -ne 1 ]; then
  echo "$0 missing PIPELINE_RUN_CACHE argument"
  exit 1
fi

PIPELINE_RUN_CACHE="$1"

# Sources the run cache
. $PIPELINE_RUN_CACHE

python -m opal.evaluate \
--model_path "$MODEL_PATH" \
--dataset_name "$DATASET_NAME" \
|| exit 1
