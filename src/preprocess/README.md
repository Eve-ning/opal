# Preprocessing Scripts

These holds all preprocessing scripts and preprocessing dependencies for opal.

- `osu-data-docker`: Git submodule to spins up osu! Data as a MySQL container for preprocessing.
- `osu-data-docker.env`: Environment variables for the osu! Data container.
- `compute_opal_svness.DockerFile`: A Dockerfile that builds a container that runs the `compute_opal_svness.py` script.
- `compute_opal_svness.py`: A script that computes the SVness of a beatmap and stores it in the database.
- `compute_opal_tables_1.sql` and `compute_opal_tables_2.sql`: SQL scripts that create the tables needed for preprocessing.
- `export_opal_active_scores.sql`: SQL script that exports the active scores (dataset) to a CSV file.
- `preprocess.sh`: Entry point for preprocessing. This script runs all preprocessing scripts in the correct order.

## `preprocess.sh`

1. We'll spin up osu! Data on Docker, pulling all necessary data dumps from https://data.ppy.sh, prepopulating a MySQL container.
2. Then, `compute_opal_tables_1` will create the necessary first set of tables for preprocessing.
3. Next, `compute_opal_svness` will compute the SVness of each beatmap and store it in the database.
4. `compute_opal_tables_2` will create the necessary second set of tables for preprocessing, which depends on the SVness.
5. Finally, `export_opal_active_scores` will export the active scores (dataset) to a CSV file.
6. Also the Docker container will be stopped.

