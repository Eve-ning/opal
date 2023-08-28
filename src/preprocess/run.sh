#!/usr/bin/env bash

compute_opal_tables() {
  if ! docker ps | grep -q osu.mysql; then
    echo -e "\e[osu.mysql is not running.\e[0m"
    exit 1
  fi

  # We'll check if opal_active_scores, the table to train opal, is present
  # If not, then we run the .sql to generate it, which takes a few minutes.
  if ! docker exec osu.mysql mysql \
    -u root --password=p@ssw0rd1 -D osu \
    -e 'SELECT * FROM opal_active_scores LIMIT 1;' >>/dev/null 2>&1; then
    echo -e "\e[33mopal_active_scores is absent, creating opal tables (~10mins)\e[0m"

    echo -e "\e[33m(1/3) Computing 1st opal table set\e[0m"
    docker exec -i osu.mysql mysql -u root --password=p@ssw0rd1 -D osu <./compute_opal_tables_1.sql

    # We define a network between the osu.mysql and our svness calc
    if [ -z "$(docker network ls -f name=mysql-net -q)" ]; then
      echo -e "\e[33mmysql-net not found, creating mysql-net\e[0m"
      docker network create mysql-net
    fi

    docker network connect mysql-net osu.mysql >>/dev/null 2>&1
    echo -e "\e[33(2/3) Computing SV-ness\e[0m"
    docker build -t compute_opal_svness -f ./compute_opal_svness.Dockerfile .
    docker run --rm --network mysql-net --mount type=bind,source="/var/lib/osu/",target="/var/lib/osu/" \
      compute_opal_svness

    echo -e "\e[33m(3/3) Computing 2nd opal table set\e[0m"
    docker exec -i osu.mysql mysql -u root --password=p@ssw0rd1 -D osu <./compute_opal_tables_2.sql

    echo -e "\e[33mopal tables created\e[0m"
  fi
}

cd_to_script() {
  # We allow this script to be run from any location, we just force it to cd here before executing the rest.
  PROJ_DIR="$(dirname "$(realpath "$0")")"
  cd "$PROJ_DIR" || exit 1
}

docker_compose_up() {
  # We have to run compose WITHIN the submod, so it can detect dependent sh files.
  cd osu-data-docker/ || exit 1
  docker compose \
    --profile files \
    -f docker-compose.yml \
    --env-file .env \
    --env-file ../osu-data-docker.env \
    up --wait --build
  cd .. || exit 1
}

docker_compose_down() {
  cd osu-data-docker/ || exit 1
  docker compose \
    --profile files \
    -f docker-compose.yml \
    --env-file .env \
    --env-file ../osu-data-docker.env \
    stop
  cd .. || exit 1
}

# Exports the opal_active_scores table to a csv file.
export_opal_active_scores() {
  docker exec -i osu.mysql mysql -u root --password=p@ssw0rd1 -D osu \
    <./export_opal_active_scores.sql |
    sed 's/\t/,/g' \
      >"$1"
}

PIPELINE_RUN_CACHE="$1"
echo "PIPELINE_RUN_CACHE=$PIPELINE_RUN_CACHE"
source "$PIPELINE_RUN_CACHE"

# Ensure that the required variables are set
[ -z "$DB_URL" ] && echo "DB_URL not set" && exit 1
[ -z "$FILES_URL" ] && echo "FILES_URL not set" && exit 1

export DB_URL FILES_URL

DATASET_NAME=$(basename "$DB_URL" .tar.bz2)_$(date +"%Y%m%d%H%M%S").csv
DATASET_PATH=../datasets/$DATASET_NAME

echo DATASET_NAME="$DATASET_NAME" >> "$PIPELINE_RUN_CACHE"

cd_to_script
docker_compose_up
compute_opal_tables
export_opal_active_scores "$DATASET_PATH"
docker_compose_down
