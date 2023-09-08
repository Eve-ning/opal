cd "$(dirname "$0")" || exit 1

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

# Find the newest .ckpt file in ../opal/models all subdirs and get its path
MODEL_PATH=$(find ../opal/models -type f -name "*.ckpt" -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" ")
# Replace .. with /src
MODEL_PATH=${MODEL_PATH/..\//\/src\/}
echo -e "\e[34mMODEL_PATH:\e[0m" "$MODEL_PATH"

env_add "$ENV_FILE_PATH" "MODEL_PATH" "$MODEL_PATH"
