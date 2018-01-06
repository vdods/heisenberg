#!/bin/bash -e

# This dir needs to exactly match the one in generate.sh
BASE_DIR="NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem"
FIGURE_DIR="$BASE_DIR/Figure4"
OUTPUT_DIR="$FIGURE_DIR/generated-data"

if [ ! -d "$BASE_DIR" ]; then
    echo "This script must be run from the heisenberg project root (i.e. the one containing the files .gitignore and LICENSE.md)."
    exit -1
fi

mkdir -p "$OUTPUT_DIR"

# Compute the objective function on the uniform grid of (p_theta, J) values.  This program uses internal parallel
# processing, and will use as many processor cores as you have.
/usr/bin/time python3 -m heisenberg.sample --dt=0.01 --max-time=100 --embedding-dimension=2 --embedding-solution-sheet-index=0 --samples-dir="$OUTPUT_DIR" --seed=54325438 --sampling-domain=[[-0.05,0.05],[0.0,0.4]] --sampling-type=ordered --sampling-range=[25,100]
