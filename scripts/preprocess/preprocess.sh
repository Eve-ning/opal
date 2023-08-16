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
    echo -e "\e[33mopal_active_scores is absent, creating opal tables\e[0m"
    echo -e "\e[32mThis will take around 10 minutes\e[0m"
    docker exec -i osu.mysql mysql -u root --password=p@ssw0rd1 -D osu <./create_opal_tables.sql
    echo -e "\e[33mopal tables created\e[0m"
  fi
}

compute_visual_complexity() {
  docker build -t compute_visual_complexity -f ./compute_visual_complexity.Dockerfile .
  docker run --rm compute_visual_complexity
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

cd_to_script
docker_compose_up
create_opal_tables
compute_opal_svness
docker_compose_down
