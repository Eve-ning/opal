#!/bin/bash

compute_opal_tables() {
  if ! docker ps | grep -q osu.mysql; then
    echo -e "\e[osu.mysql is not running.\e[0m"
    exit 1
  fi

  # We'll check if opal_active_scores, the table to train opal, is present
  # If not, then we run the .sql to generate it, which takes a few minutes.
  if docker exec osu.mysql mysql \
    -u root --password=p@ssw0rd1 -D osu \
    -e 'SELECT * FROM opal_active_scores LIMIT 1;' >>/dev/null 2>&1; then
    echo -e "\e[33mopal_active_scores is present, skip creating opal tables\e[0m"
  else
    echo -e "\e[33mopal_active_scores is absent, creating opal tables (~10mins)\e[0m"

    echo -e "\e[33m(1/3) Computing 1st opal table set\e[0m"
    docker exec -i osu.mysql mysql -u root --password=p@ssw0rd1 -D osu <./compute_opal_tables_1.sql

    # We define a network between the osu.mysql and our svness calc
    if [ -z "$(docker network ls -f name=mysql-net -q)" ]; then
      echo -e "\e[33mmysql-net not found, creating mysql-net\e[0m"
      docker network create mysql-net
      docker network connect mysql-net osu.mysql
    fi
    echo -e "\e[33(2/3) Computing SV-ness\e[0m"
    docker build -t compute_opal_svness -f ./compute_opal_svness.Dockerfile .
    docker run --rm --network mysql-net --mount type=bind,source="/var/lib/osu/",target="/var/lib/osu/" \
      compute_opal_svness

    echo -e "\e[33m(3/3) Computing 2nd opal table set\e[0m"
    docker exec -i osu.mysql mysql -u root --password=p@ssw0rd1 -D osu <./compute_opal_tables_1.sql

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
  echo "Moving to osu-data-docker"
  echo "Docker Compose Up"
  docker compose \
    --profile files \
    -f docker-compose.yml \
    --env-file .env \
    --env-file ../osu-data-docker.env \
    up --wait --build
  cd .. || exit 1
  echo "Moving to out of osu-data-docker"
}

docker_compose_down() {
  cd osu-data-docker/ || exit 1
  echo "Moving to osu-data-docker"
  echo "Docker Compose Stop"
  docker compose \
    --profile files \
    -f docker-compose.yml \
    --env-file .env \
    --env-file ../osu-data-docker.env \
    stop
  cd .. || exit 1
  echo "Moving to out of osu-data-docker"
}

# Exports the opal_active_scores table to a csv file.
export_opal_active_scores() {
  docker exec -i osu.mysql mysql -u root --password=p@ssw0rd1 -D osu \
    <./export_opal_active_scores.sql |
    sed 's/\t/,/g' \
      >"$1"
}

export DB_URL=https://data.ppy.sh/2023_08_01_performance_mania_top_1000.tar.bz2
export FILES_URL=https://data.ppy.sh/2023_08_01_osu_files.tar.bz2
EXPORT_CSV=../datasets/"$(basename "$DB_URL" .tar.bz2)"_$(date +"%Y%m%d%H%M%S").csv
export EXPORT_CSV

cd_to_script
docker_compose_up
compute_opal_tables
export_opal_active_scores "$EXPORT_CSV"
docker_compose_down
