#!/bin/bash -e

# This dir needs to exactly match the one in generate.sh
BASE_DIR="Numerical Methods and Closed Orbits in the Kepler–Heisenberg Problem"
FIGURE_DIR="$BASE_DIR/Figure4"
OUTPUT_DIR="$FIGURE_DIR/generated-data"

if [ ! -d "$BASE_DIR" ]; then
    echo "This script must be run from the heisenberg project root (i.e. the one containing the files .gitignore and LICENSE.md)."
    exit -1
fi

mkdir -p "$OUTPUT_DIR"

# Plot the generated samples.
/usr/bin/time --verbose python3 -m heisenberg.plot_samples --plot-type=pdf --samples-dir="$OUTPUT_DIR"
