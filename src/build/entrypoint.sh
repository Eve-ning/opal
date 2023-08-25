#!/bin/bash
git config --global --add safe.directory /opal
cp ../README.md ./README.md
poetry build
rm ./README.md
echo "Build complete"

# Publish if PYPI_TOKEN is set
