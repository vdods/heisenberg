#!/bin/bash -e

BASE_DIR="SelfSimilarityInTheKeplerHeisenbergProblem"
GENERATED_DATA_DIR="$BASE_DIR/generated-data"

if [ ! -d "$BASE_DIR" ]; then
    echo "This script must be run from the heisenberg project root (i.e. the one containing the files .gitignore and LICENSE.md)."
    exit -1
fi

echo "Generating all results.  Note that this will take several minutes..."

rm -rf "$GENERATED_DATA_DIR"
mkdir -p "$GENERATED_DATA_DIR"

python3.6 -m heisenberg.self_similar
