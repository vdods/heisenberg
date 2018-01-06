#!/bin/bash -e

BASE_DIR="NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem"
FIGURE_DIR="$BASE_DIR/Figure6"
GENERATED_DATA_DIR="$FIGURE_DIR/generated-data"
OUTPUT_DIR="$GENERATED_DATA_DIR/plots" # This is the directory that the *.plot_commands specifies

if [ ! -d "$BASE_DIR" ]; then
    echo "This script must be run from the heisenberg project root (i.e. the one containing the files .gitignore and LICENSE.md)."
    exit -1
fi

mkdir -p "$OUTPUT_DIR"

echo "Generating local minimum plots in parallel..."
echo "Generated files will appear in the directory \"$OUTPUT_DIR\""
cat "$GENERATED_DATA_DIR/sample_v.count:1000.plot_commands" | /usr/bin/time parallel 2>&1 | tee "$OUTPUT_DIR/log.txt"
