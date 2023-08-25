# Change directory to current script directory
cd "$(dirname "$(realpath "$0")")" || exit 1

# Create unique pipeline run id
PIPELINE_RUN_CACHE=.pipeline_cache/"$(date +%s).env"
mkdir -p "$(dirname "$PIPELINE_RUN_CACHE")"

{
  echo DB_URL=https://github.com/Eve-ning/opal/raw/pipeline-automation/sample.tar.bz2
  echo FILES_URL=https://github.com/Eve-ning/opal/raw/pipeline-automation/sample_files.tar.bz2
  echo MODEL_NAME="${MODEL_NAME="2023.8.4b"}"
} >>"$PIPELINE_RUN_CACHE"
source "$PIPELINE_RUN_CACHE"

DATASET_NAME=$(basename "$DB_URL" .tar.bz2)_$(date +"%Y%m%d%H%M%S").csv
export DB_URL FILES_URL DATASET_NAME

echo "Executing Preprocessing Step"
export MIN_SCORES_PER_MID=0
export MIN_SCORES_PER_UID=0
docker compose \
--profile files \
-f preprocess/docker-compose.yml \
--env-file preprocess/osu-data-docker/.env \
up --build

exit 0

source "$PIPELINE_RUN_CACHE"
[ -z "$DATASET_NAME" ] && echo "DATASET_NAME not returned by preprocess" && exit 1

exit 0

echo "Training Model"
PIPELINE_RUN_CACHE="$PIPELINE_RUN_CACHE" docker compose -f train/docker-compose.yml up --build

source "$PIPELINE_RUN_CACHE"
[ -z "$MODEL_PATH" ] && echo "MODEL_PATH not returned by train" && exit 1

echo "Evaluating Model"
PIPELINE_RUN_CACHE="$PIPELINE_RUN_CACHE" docker compose -f evaluate/docker-compose.yml up --build

echo "Publishing Model"
PIPELINE_RUN_CACHE="$PIPELINE_RUN_CACHE" docker compose -f build/docker-compose.yml up --build
