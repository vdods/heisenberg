#!/bin/bash -e

# This dir needs to exactly match the one in generate.sh
BASE_DIR="Numerical Methods and Closed Orbits in the Kepler–Heisenberg Problem"
FIGURE_DIR="$BASE_DIR/Figure3"
OUTPUT_DIR="$FIGURE_DIR/generated-data"

if [ ! -d "$BASE_DIR" ]; then
    echo "This script must be run from the heisenberg project root (i.e. the one containing the files .gitignore and LICENSE.md)."
    exit -1
fi

mkdir -p "$OUTPUT_DIR"

# The 1/2 orbit
/usr/bin/time --verbose python3 -m heisenberg.plot --dt=0.003 --max-time=1.679e+01 --embedding-dimension=1 --embedding-solution-sheet-index=0 --plot-type=pdf --initial-preimage=[1.64380951372436468e-01] --output-dir="$OUTPUT_DIR" --quantities-to-plot="x,y;t,z" --use-terse-plot-titles --plot-size=3
