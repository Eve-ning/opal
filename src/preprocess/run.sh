# Usage: ./run.sh ENV_FILE_PATH
#
# The ENV_FILE_PATH is the path to the .env file to source and add variables to.
# The .env file should contain the following variables:
# - SR_MIN
# - SR_MAX
# - ACC_MIN
# - ACC_MAX
# - MIN_SCORES_PER_MID
# - MIN_SCORES_PER_UID
# - FILES_DIR
# - DB_NAME
# - DB_USERNAME
# - DB_PASSWORD
# - DB_HOST
# - DB_PORT
# - MAX_SVNESS
# - DATASET_NAME

cd "$(dirname "$0")" || exit 1

create_dataset_name() {
  SQL_HASH=$(find . -type f -name "*.sql" -exec cat {} \; | sort | md5sum | cut -c1-8)
  DATASET_HASH=$(echo "$SR_MIN$SR_MAX$ACC_MIN$ACC_MAX$MIN_SCORES_PER_MID$MIN_SCORES_PER_UID$MAX_SVNESS" | md5sum | cut -c1-8)
  DATASET_NAME="$(basename "$DB_URL" .tar.bz2)_${DATASET_HASH}_${SQL_HASH}.csv"

  # echo with no new line
  echo -e "\e[34mSQL_HASH:\e[0m $SQL_HASH "
  echo -e "\e[34mDATASET_HASH:\e[0m $DATASET_HASH "
  echo -e "\e[34mDB_NAME:\e[0m $DB_NAME"
  echo -e "\e[34mDATASET_NAME:\e[0m" "$DATASET_NAME"

  export DATASET_NAME
  env_add "$FILE_PATH" "DATASET_NAME" "$DATASET_NAME"
}

# Check if the .env file path is the first argument and source it
if [ -f "$1" ]; then
  FILE_PATH="$1"
  set -a
  source "$FILE_PATH"
  set +a
else
  echo "Usage: ./run.sh [ENV_FILE_PATH]"
  echo "The ENV_FILE_PATH must be relative to this script."
  exit 1
fi

# Load utils for envdotsub and check_env_set
. ../utils.sh
create_dataset_name
export FILES_DIR="/var/lib/osu/osu.files/$(basename "$FILES_URL" .tar.bz2)/"

# Check if all necessary env vars are set
check_env_set docker-compose.yml || exit 1
check_env_set osu-data-docker/docker-compose.yml || exit 1

# Check if the dataset already exists
if [ -f "../datasets/$DATASET_NAME" ]; then
  echo -e "\e[33mDataset already exists: $DATASET_NAME\e[0m"
  exit 0
fi

envdotsub docker-compose.yml || exit 1
envdotsub osu-data-docker/docker-compose.yml || exit 1

# Without -d, this script will hang until the docker compose process is killed
# To also include compose stop, we'll peek at the dataset file and wait for it to be created
docker compose --profile files -f .docker-compose.yml up --build -d >preprocess.log 2>&1 &

while [ ! -f "../datasets/$DATASET_NAME" ]; do
  echo "Waiting for dataset to be created... (Showing most recent log)"
  tail -n 3 preprocess.log
  sleep 10
done

docker compose --profile files -f .docker-compose.yml stop || exit 1

echo "Preprocessing complete. Dataset: $DATASET_NAME"
