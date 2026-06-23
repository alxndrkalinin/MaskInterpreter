#!/bin/bash

# Define the base directory
base_dir="${SEARCH_PATH}"

# Use find to locate all *.csv files and sed to replace the text
find "$base_dir" -type f -name "*.csv" -exec sed -i 's|***|'"$REPLACE_TO"'|g' {} +

echo "Replacement complete."
