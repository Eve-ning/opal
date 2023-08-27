# Change directory to current script directory
cd "$(dirname "$(realpath "$0")")" || exit 1

# Create unique pipeline run id
PIPELINE_RUN_CACHE=.pipeline_cache/"$(date +%s).env"
mkdir -p "$(dirname "$PIPELINE_RUN_CACHE")"

cat << EOF >>"$PIPELINE_RUN_CACHE"
DB_URL=https://github.com/Eve-ning/opal/raw/pipeline-automation/rsc/sample.tar.bz2
FILES_URL=https://github.com/Eve-ning/opal/raw/pipeline-automation/rsc/sample_files.tar.bz2
FILES_DIR=/var/lib/osu/osu.files/sample_files/
MODEL_NAME="${MODEL_NAME="2023.8.4b"}"
DATASET_NAME="$(basename "$DB_URL" .tar.bz2)_$(date +"%Y%m%d%H%M%S").csv"
DB_NAME=osu
DB_USERNAME=root
DB_PASSWORD=p@ssw0rd1
DB_HOST=osu.mysql
DB_PORT=3307
EOF


source "$PIPELINE_RUN_CACHE"
export PIPELINE_RUN_CACHE DB_URL FILES_URL FILES_DIR MODEL_NAME DATASET_NAME

echo "Preprocessing"
export MIN_SCORES_PER_MID=0
export MIN_SCORES_PER_UID=0
docker compose \
--profile files \
-f preprocess/docker-compose.yml \
--env-file preprocess/osu-data-docker/.env \
up --build || exit 1

source "$PIPELINE_RUN_CACHE"
[ -z "$DATASET_NAME" ] && echo "DATASET_NAME not returned by preprocess" && exit 1

echo "Training Model"
docker compose \
-f train/docker-compose.yml \
up --build || exit 1

source "$PIPELINE_RUN_CACHE"
[ -z "$MODEL_PATH" ] && echo "MODEL_PATH not returned by train" && exit 1

echo "Evaluating Model"
docker compose \
-f evaluate/docker-compose.yml \
up --build || exit 1

echo "Publishing Model"
docker compose \
-f build/docker-compose.yml \
up --build || exit 1
