#!/bin/bash

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -b)
        BRANCH="$2"
        shift
        shift
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

# Check if branch is provided
if [ -z "$BRANCH" ]; then
    echo "Please provide a branch name using --m-d-branch"
    exit 1
fi

# Create temporary directory
TEMP_DIR="maze-dataset_test"

# Copy pyproject.toml and poetry.lock to temporary directory
# cp pyproject.toml poetry.lock "$TEMP_DIR/"
cd "$TEMP_DIR"
echo "Exiting the poetry shell if inside one"; deactivate || echo "Not already in a poetry shell"
poetry env info
# poetry show maze-dataset
# poetry remove maze-dataset
# poetry add git+https://github.com/understanding-search/maze-dataset.git#$BRANCH
echo Installed maze-dataset#$BRANCH
# poetry show maze-dataset
POSIX_PATH=$(echo $(poetry env info --path)| sed 's/\\/\//g')
echo "POSIX_PATH = $POSIX_PATH"
echo "trying Linux on Windows" && source $(poetry env info --path)\\Scripts\\activate.ps1 || echo "trying linux activate" && source $POSIX_PATH/bin/activate
cd ..
poetry env info
