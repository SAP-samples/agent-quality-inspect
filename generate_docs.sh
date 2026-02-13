#!/bin/bash

# The following steps assume that sphinx-quickstart files are already generated and updated already

# Initialize EXCLUDES as an empty array
EXCLUDES=()

# Read lines from the file and append to the array
while IFS= read -r line || [[ -n "$line" ]]; do
  EXCLUDES+=("$line")
done < module_to_exclude.txt

# Print all excludes
printf 'Exclude: %s\n' "${EXCLUDES[@]}"

sphinx-apidoc -o docs src/agent_inspect "${EXCLUDES[@]}"

# set maxdepth to 1 in the generated docs/*.rst files
find docs -name '*.rst' -exec sed -i '' -e '/^.. toctree::/{
    n
    s/.*/   :maxdepth: 1/
}' {} +


cd docs

make clean
make html

cd ..