#!/bin/bash -e

# This dir needs to exactly match the one in generate.sh
BASE_DIR="Numerical Methods and Closed Orbits in the Kepler–Heisenberg Problem"
FIGURE_DIR="$BASE_DIR/Figure1"
OUTPUT_DIR="$FIGURE_DIR/generated-data"

if [ ! -d "$BASE_DIR" ]; then
    echo "This script must be run from the heisenberg project root (i.e. the one containing the files .gitignore and LICENSE.md)."
    exit -1
fi

mkdir -p "$OUTPUT_DIR"

# The 5/9 orbit
/usr/bin/time --verbose python3 -m heisenberg.plot --dt=0.003 --max-time=80.74300295598482 --initial-preimage=[0.14693671554021892] --embedding-dimension=1 --embedding-solution-sheet-index=0 --output-dir="$OUTPUT_DIR" --disable-plot-decoration --cut-off-initial-curve-tail --quantities-to-plot=x,y --plot-type=pdf
