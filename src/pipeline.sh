# Change directory to current script directory
cd "$(dirname "$(realpath "$0")")" || exit 1

# Create unique pipeline run id
PIPELINE_RUN_CACHE=.pipeline_cache/"$(date +%s).env"
mkdir -p "$(dirname "$PIPELINE_RUN_CACHE")"

{
  echo YYYY_MM="${YYYY_MM:="2023_08"}"
  echo DATASET="${DATASET="top_1000"}"
  echo MODEL_NAME="${MODEL_NAME="2023.8.4b"}"
} >>"$PIPELINE_RUN_CACHE"

echo "Executing Preprocessing Step"
./preprocess/run.sh "$PIPELINE_RUN_CACHE" || exit 1

source "$PIPELINE_RUN_CACHE"
[ -z "$DATASET_NAME" ] && echo "DATASET_NAME not returned by preprocess" && exit 1

echo "Training Model"
PIPELINE_RUN_CACHE="$PIPELINE_RUN_CACHE" docker compose -f train/docker-compose.yml up --build

source "$PIPELINE_RUN_CACHE"
[ -z "$MODEL_PATH" ] && echo "MODEL_PATH not returned by train" && exit 1

echo "Evaluating Model"
PIPELINE_RUN_CACHE="$PIPELINE_RUN_CACHE" docker compose -f evaluate/docker-compose.yml up --build

# Substitutes version = "..." with version = "$MODEL_NAME"
sed -i "s|version = .*|version = \"$MODEL_NAME\"|" ./pyproject.toml
grep -q "<VERSION>" pyproject.toml && echo "Version failed to substitute" && exit 1

# Substitutes include = ["..."] with include = ["opal/$MODEL_NAME/*"]
sed -i "s|include = \[.*\]|include = [\"opal/$MODEL_NAME/*\"]|" ./pyproject.toml
grep -q "<MODEL_PATH>" pyproject.toml && echo "Model path failed to substitute" && exit 1

echo "Publishing Model"
docker compose -f build/docker-compose.yml up --build
