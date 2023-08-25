#!/bin/bash
# Ensure that one argument is passed
if [ "$#" -ne 1 ]; then
  echo "$0 missing PIPELINE_RUN_CACHE argument"
  exit 1
fi

PIPELINE_RUN_CACHE="$1"

# Sources the run cache
. $PIPELINE_RUN_CACHE

git config --global --add safe.directory /opal
cp ../README.md ./README.md || exit 1

# Substitutes version = "..." with version = "$MODEL_NAME"
sed -i "s|version = .*|version = \"$MODEL_NAME\"|" ./pyproject.toml || exit 1
grep -q "<VERSION>" ./pyproject.toml && echo "Version failed to substitute" && exit 1

# Substitutes include = ["..."] with include = ["opal/$MODEL_NAME/*"]
sed -i "s|include = \[.*\]|include = [\"opal/$MODEL_NAME/*\"]|" ./pyproject.toml || exit 1
grep -q "<MODEL_PATH>" ./pyproject.toml && echo "Model path failed to substitute" && exit 1

poetry build || exit 1
rm ./README.md
echo "Build complete"

# Publish if PYPI_TOKEN is set
if [ -z "$PYPI_TOKEN" ]; then
  echo -e "\e[33mPYPI_TOKEN not set, skipping publish\e[0m"
  exit 1
fi
poetry config pypi-token.pypi "$PYPI_TOKEN"
poetry publish --build