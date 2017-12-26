#!/bin/bash -e

# This dir needs to exactly match the one in generate.sh
BASE_DIR="Numerical Methods and Closed Orbits in the Kepler–Heisenberg Problem"
FIGURE_DIR="$BASE_DIR/Figure5"
OUTPUT_DIR="$FIGURE_DIR/generated-data"

if [ ! -d "$BASE_DIR" ]; then
    echo "This script must be run from the heisenberg project root (i.e. the one containing the files .gitignore and LICENSE.md)."
    exit -1
fi

mkdir -p "$OUTPUT_DIR"

# A quasi-periodic orbit having near-6-fold symmetry
/usr/bin/time --verbose python3 -m heisenberg.plot --dt=1.0e-2 --max-time=600 --embedding-dimension=2 --embedding-solution-sheet-index=1 --initial-preimage=[4.35797440367095557e-02,3.48993491514963006e-01] --quantities-to-plot="x,y;t,z" --plot-type=pdf --use-terse-plot-titles --plot-size=3 --output-dir="$OUTPUT_DIR"
