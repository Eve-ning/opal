# Change directory to current script directory
cd "$(dirname "$(realpath "$0")")" || exit 1

# Create unique pipeline run id
PIPELINE_RUN_CACHE=.pipeline_cache/"$(date +%s).env"
mkdir -p "$(dirname "$PIPELINE_RUN_CACHE")"

DB_URL=https://github.com/Eve-ning/opal/raw/pipeline-automation/rsc/sample.tar.bz2
FILES_URL=https://github.com/Eve-ning/opal/raw/pipeline-automation/rsc/sample_files.tar.bz2
cat << EOF >>"$PIPELINE_RUN_CACHE"
PIPELINE_RUN_CACHE=$PIPELINE_RUN_CACHE
DB_URL=$DB_URL
FILES_URL=$FILES_URL
FILES_DIR=/var/lib/osu/osu.files/$(basename "$FILES_URL" .tar.bz2)/
MODEL_NAME=2023.8.4b
DATASET_NAME=$(basename "$DB_URL" .tar.bz2)_$(date +"%Y%m%d%H%M%S").csv
DB_NAME=osu
DB_USERNAME=root
DB_PASSWORD=p@ssw0rd1
DB_HOST=osu.mysql
DB_PORT=3307
SR_MIN=2
SR_MAX=15
ACC_MIN=0.85
ACC_MAX=1.0
MIN_SCORES_PER_MID=0
MIN_SCORES_PER_UID=0
MAX_SVNESS=0.05
EOF

set -a
source "$PIPELINE_RUN_CACHE"
set +a

echo "Preprocessing"
docker compose \
--profile files \
-f preprocess/docker-compose.yml \
--env-file preprocess/osu-data-docker/.env \
up -d > output.log 2>&1 &

# Wait until the dataset in ./datasets/$DATASET_NAME is created
while [ ! -f "./datasets/$DATASET_NAME" ]; do
  echo "Waiting for dataset to be created... (Showing most recent log)"
  tail -n 1 output.log
  sleep 10
done

docker compose \
--profile files \
-f preprocess/docker-compose.yml \
--env-file preprocess/osu-data-docker/.env \
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
