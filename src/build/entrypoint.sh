#!/usr/bin/env bash

# Ensure all variables are set
: "${MODEL_NAME:?MODEL_NAME not set}"

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