#!/bin/bash


# Check that opal.mysql is running
if ! docker ps | grep -q opal.mysql; then
  echo -e "\e[31mopal.mysql is not running, please run it first\e[0m"
  exit 1
fi

echo -e "\e[33mCheck that osu.mysql has completed initialization\e[0m"
while ! docker logs opal.mysql 2>&1 | grep -q "MySQL init process done. Ready for start up."; do
  echo -en "\e[33mWaiting for initialization... \e[0m"
  echo -e "\e[34m Log: $(docker logs opal.mysql 2>&1 | tail -1)\e[0m"
  sleep 5
done
echo -e "\e[32mosu.mysql has completed initialization\e[0m"

# We'll check if opal_active_scores, the table to train opal, is present
# If not, then we run the .sql to generate it, which takes a few minutes.
if docker exec opal.mysql mysql \
   -u root --password=p@ssw0rd1 -D osu \
   -e 'SELECT * FROM opal_active_scores LIMIT 1;' >> /dev/null 2>&1; then
  echo -e "\e[33mopal_active_scores is present, skip creating opal tables\e[0m"
else
  echo -e "\e[33mopal_active_scores is absent, creating opal tables\e[0m"
  echo -e "\e[32mThis will take around 10 minutes\e[0m"
  docker exec -i opal.mysql mysql -u root --password=p@ssw0rd1 -D osu < ./create_opal_tables.sql
  echo -e "\e[33mopal tables created\e[0m"
fi

exit 0
