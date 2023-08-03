# Ensure that caller is in the directory of the script.sh
#    Retrieve the absolute path (realpath) of the script, then gets the directory (dirname)
#    Then CD into it.
PROJ_DIR="$(dirname "$(realpath "$0")")"
cd "$PROJ_DIR" || exit 1

docker compose \
  --project-directory ../ \
  --profile files \
  -f ./docker-compose.yml \
  --env-file ./.env \
  up
