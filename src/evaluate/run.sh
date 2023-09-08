# Check if the .env file path is the first argument and source it
if [ -f "$1" ]; then
  ENV_FILE_PATH="$1"
  export ENV_FILE_PATH
  set -a
  source "$ENV_FILE_PATH"
  set +a
else
  echo "Usage: ./run.sh [ENV_FILE_PATH]"
  exit 1
fi

# Load utils for envdotsub and check_env_set
. ../utils.sh
check_env_set docker-compose.yml || exit 1

envdotsub docker-compose.yml
docker compose -f .docker-compose.yml up --build || exit 1
