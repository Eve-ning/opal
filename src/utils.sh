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

# Usage: check_env_set [FILE_PATH]
# Check if all env vars are set in the file.
# Exit 1 if any variable is found unset.
check_env_set() {
  # Check if the first argument is the docker-compose file else print usage
  if [ -z "$1" ]; then
    echo "Usage: check_env_set [FILE_PATH]"
    exit 1
  fi

  # This matches all cases of ${VAR} and extracts VAR
  vars=$(grep -oP '\$\{\K[^}]*' "$1")

  # Iterate over extracted variables
  ANY_UNSET=false
  for var in $vars; do
    if [ -z "${!var}" ]; then
      echo -e "\e[31mVariable $var is not set\e[0m"
      ANY_UNSET=true
    fi
  done

  # Exit 1 if any variable is unset
  if $ANY_UNSET; then
    echo -e "\e[31mOne or more variables are not set. See above for details.\e[0m"
    exit 1
  fi
}

# Usage: env_add [FILE_PATH] [ENV_VAR_NAME] [ENV_VAR_VALUE]
# Append an env var to the file if it doesn't exist, else replace it.
env_add() {
  if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: env_add [FILE_PATH] [ENV_VAR_NAME] [ENV_VAR_VALUE]"
    exit 1
  fi

  # Assign arguments to variables
  FILE_PATH="$1"
  ENV_VAR_NAME="$2"
  ENV_VAR_VALUE="$3"

  if grep -q "$ENV_VAR_NAME" "$FILE_PATH"; then
    # If the variable already exists, replace it
    sed -i "s|$ENV_VAR_NAME=.*|$ENV_VAR_NAME=$ENV_VAR_VALUE|g" "$1" || exit 1
  else
    # Else, append it to the end of the file
    echo "$ENV_VAR_NAME=$ENV_VAR_VALUE" >>"$FILE_PATH"
  fi
}
