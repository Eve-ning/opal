#!/bin/bash

# We'll check if opal_active_scores, the table to train opal, is present
# If not, then we run the .sql to generate it, which takes a few minutes.
if docker exec osu.mysql mysql \
   -u root --password=p@ssw0rd1 -D osu \
   -e 'SELECT * FROM opal_active_scores LIMIT 1;' >> /dev/null 2>&1; then
  echo -e "\e[33mopal_active_scores is present, skip creating opal tables\e[0m"
else
  echo -e "\e[33mopal_active_scores is absent, creating opal tables\e[0m"
  echo -e "\e[32This will take around 10 minutes\e[0m"
  docker exec -i osu.mysql mysql -u root --password=p@ssw0rd1 -D osu < ./create_opal_tables.sql
  echo -e "\e[33mopal tables created\e[0m"
fi

exit 0
