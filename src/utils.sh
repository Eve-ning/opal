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
    else
      echo "$var: ${!var}"
    fi
  done

  # Exit 1 if any variable is unset
  if $ANY_UNSET; then
    echo -e "\e[31mOne or more variables are not set. See above for details.\e[0m"
    exit 1
  fi
}

