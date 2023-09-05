# Usage: envdotsub [FILE_PATH]
# Substitute env vars in [FILE_PATH] and output to .[FILE_PATH]
envdotsub() {
  if [ -z "$1" ]; then
    echo "Usage: envdotsub [FILE_PATH]"
    exit 1
  fi
  filename=$(basename "$1")
  dir=$(dirname "$1")
  dotfile=".$filename"
  dotfilepath="$dir/$dotfile"
  envsubst <"$1" >"$dotfilepath" || exit 1
}
