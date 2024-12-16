#!/usr/bin/env bash

ORIGINAL_PARAMS=("$@")

INCLUDE_FILE_TYPES=()
SHOW_STRUCTURE=false
INCLUDE_DIRS=()

while (( "$#" )); do
  case "$1" in
    --include-file-types)
      shift
      IFS=' ' read -r -a INCLUDE_FILE_TYPES <<< "$1"
      shift
      ;;
    --show-structure)
      SHOW_STRUCTURE=true
      shift
      ;;
    --include-dirs)
      shift
      IFS=' ' read -r -a INCLUDE_DIRS <<< "$1"
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

if [ ${#INCLUDE_DIRS[@]} -eq 0 ]; then
  SEARCH_PATHS=(".")
else
  SEARCH_PATHS=("${INCLUDE_DIRS[@]}")
fi

EXCLUDE_PATHS=(
  -not -path "*/venv/*"
  -not -path "*/.git/*"
  -not -path "*/__pycache__/*"
  -not -path "*/env/*"
  -not -path "*/site-packages/*"
)

FIND_EXPRS=(-name "*.py")

if [ ${#INCLUDE_FILE_TYPES[@]} -gt 0 ]; then
  NEW_FIND_EXPRS=()
  for ft in "${INCLUDE_FILE_TYPES[@]}"; do
    NEW_FIND_EXPRS+=(-o -name "$ft")
  done
  FIND_EXPRS=( \( "${FIND_EXPRS[@]}" "${NEW_FIND_EXPRS[@]}" \) )
else
  FIND_EXPRS=( \( "${FIND_EXPRS[@]}" \) )
fi

if $SHOW_STRUCTURE; then
  echo ">>>PROJECT STRUCTURE (excluding common dirs)<<<"
  for d in "${SEARCH_PATHS[@]}"; do
    echo "Directory: $d"
    find "$d" "${EXCLUDE_PATHS[@]}" | sed 's/^\.\///' | awk '{
      level = gsub(/\//,"/",$0);
      for (i=0; i<level; i++) printf "  ";
      print $0
    }'
  done
  echo ">>>END OF PROJECT STRUCTURE<<<"
  echo
fi

RESULT=$(
  find "${SEARCH_PATHS[@]}" -type f "${EXCLUDE_PATHS[@]}" "${FIND_EXPRS[@]}" \
    -exec sh -c '
      for f; do
        rel_path=$(realpath --relative-to="$(pwd)" "$f")
        echo ">>>FILE: $rel_path<<<"
        cat "$f" | tr "\n" " " | sed "s/  */ /g"
        echo " "
      done
    ' sh {} +
)

RESULT=$(echo "$RESULT" | sed 's/^ *//;s/ *$//')

FINAL_OUTPUT="Hello, Neural Network.

I (the user) have a problem that I'm trying to solve. I have a large repository, and I cannot provide all of it at once. To help with this, I have a script that extracts a subset of the repository for analysis. I have just run this script with the parameters shown below, and the following is the content it has copied to my clipboard.

I am now pasting this clipboard content into the chat for you to analyze. Your task: Based on the provided code, please tell me what needs to be fixed and in which files. If the provided information is not sufficient, please suggest parameters for rerunning the script (e.g., include more file types, show structure, include more directories) so that I can supply additional data.

Script parameters used:
\"$0 \"${ORIGINAL_PARAMS[*]}\"\"

Below is the code extracted by the script:

$RESULT

Available parameters for future runs:
- --include-file-types \"...\" (e.g., \"*.html Dockerfile\")
- --show-structure
- --include-dirs \"...\" (e.g., \"app templates\" quotes mandatory)
"

# Copy to clipboard using xclip
if command -v xclip &> /dev/null; then
    echo "$FINAL_OUTPUT" | xclip -sel clip
    echo "The output has been copied to your clipboard."
else
    echo "xclip not found. Please install xclip or copy the output manually."
fi

# Print to stdout as well
echo "$FINAL_OUTPUT"

# Now attempt to send keystrokes via PowerShell's SendKeys (assuming you are in WSL or have PowerShell available)
powershell.exe 'Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait("%{TAB}")'
#powershell.exe 'Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait("^v")'
#powershell.exe 'Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait("{ENTER}")'
