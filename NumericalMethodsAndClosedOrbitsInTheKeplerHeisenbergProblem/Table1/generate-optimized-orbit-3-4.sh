#!/bin/bash -e

# This dir needs to exactly match the one in generate-orbits-up-to-order-6.sh
BASE_DIR="NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem"
TABLE_DIR="$BASE_DIR/Table1"
OUTPUT_DIR="$TABLE_DIR/generated-data"

if [ ! -d "$BASE_DIR" ]; then
    echo "This script must be run from the heisenberg project root (i.e. the one containing the files .gitignore and LICENSE.md)."
    exit -1
fi

mkdir -p "$OUTPUT_DIR"

# The 3/4 orbit
python3 -m heisenberg.plot --dt=0.003 --max-time=47.22709251232482 --initial-preimage=[0.08724594805131305] --embedding-dimension=1 --embedding-solution-sheet-index=0 --output-dir="$OUTPUT_DIR" --quantities-to-plot="x,y;t,z;error(H);error(J);sqd;class-signal;objective" --plot-type=pdf --optimize-initial  --optimization-iterations=200
