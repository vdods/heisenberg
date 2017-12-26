#!/bin/bash -e

BASE_DIR="NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem"
FIGURE_DIR="$BASE_DIR/Figure6"
OUTPUT_DIR="$FIGURE_DIR/generated-data"

if [ ! -d "$BASE_DIR" ]; then
    echo "This script must be run from the heisenberg project root (i.e. the one containing the files .gitignore and LICENSE.md)."
    exit -1
fi

mkdir -p "$OUTPUT_DIR"

echo "Generating 'Figure 6' data samples in parallel..."
echo "Generated files will appear in the directory \"$OUTPUT_DIR\""
"$FIGURE_DIR/generate-samples.sh" 2>&1 | tee "$OUTPUT_DIR/log.txt"

echo "Generating 'Figure 6' plot..."
# Append to the log.txt file, so that the log will contain the output of generate-samples.sh and
# then the output of generate-plot.sh
"$FIGURE_DIR/generate-plot.sh" 2>&1 | tee --append "$OUTPUT_DIR/log.txt"
