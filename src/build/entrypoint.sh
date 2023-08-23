#!/bin/sh
git config --global --add safe.directory /app
poetry build
echo "Build complete"