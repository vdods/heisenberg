#!/bin/bash -e

BASE_DIR="NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem"
FIGURE_DIR="$BASE_DIR/Figure1"
OUTPUT_DIR="$FIGURE_DIR/generated-data"

if [ ! -d "$BASE_DIR" ]; then
    echo "This script must be run from the heisenberg project root (i.e. the one containing the files .gitignore and LICENSE.md)."
    exit -1
fi

mkdir -p "$OUTPUT_DIR"

echo "Generating 'Figure 1' plots in parallel..."
echo "Generated files will appear in the directory \"$OUTPUT_DIR\""
cat "$FIGURE_DIR/command-list.txt" | /usr/bin/time parallel 2>&1 | tee "$OUTPUT_DIR/log.txt"
