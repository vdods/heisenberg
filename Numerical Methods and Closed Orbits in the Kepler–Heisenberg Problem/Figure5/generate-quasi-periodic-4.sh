#!/bin/bash -e

# This dir needs to exactly match the one in generate-quasi-periodic.sh
BASE_DIR="Numerical Methods and Closed Orbits in the Kepler–Heisenberg Problem"
FIGURE_DIR="$BASE_DIR/Figure5"
OUTPUT_DIR="$FIGURE_DIR/generated-data"

if [ ! -d "$BASE_DIR" ]; then
    echo "This script must be run from the heisenberg project root (i.e. the one containing the files .gitignore and LICENSE.md)."
    exit -1
fi

mkdir -p "$OUTPUT_DIR"

# A quasi-periodic orbit having near-4-fold symmetry
/usr/bin/time --verbose python3 -m heisenberg.plot --dt=1.0e-4 --max-time=15 --embedding-dimension=2 --embedding-solution-sheet-index=1 --initial-preimage=[-0.0288129590255366,-0.2561400590400977] --quantities-to-plot="x,y;t,z" --plot-type=pdf --use-terse-plot-titles --plot-size=3 --output-dir="$OUTPUT_DIR"
