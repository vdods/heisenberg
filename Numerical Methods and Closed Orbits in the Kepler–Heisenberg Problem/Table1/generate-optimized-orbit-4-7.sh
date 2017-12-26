#!/bin/bash -e

# This dir needs to exactly match the one in generate-orbits-of-order-7.sh
BASE_DIR="Numerical Methods and Closed Orbits in the Kepler–Heisenberg Problem"
TABLE_DIR="$BASE_DIR/Table1"
OUTPUT_DIR="$TABLE_DIR/generated-data"

if [ ! -d "$BASE_DIR" ]; then
    echo "This script must be run from the heisenberg project root (i.e. the one containing the files .gitignore and LICENSE.md)."
    exit -1
fi

mkdir -p "$OUTPUT_DIR"

# The 4/7 orbit
python3 -m heisenberg.plot --dt=0.003 --max-time=63.9936332328723 --initial-preimage=[0.14209248839359753] --embedding-dimension=1 --embedding-solution-sheet-index=0 --output-dir="$OUTPUT_DIR" --quantities-to-plot="x,y;t,z;error(H);error(J);sqd;class-signal;objective" --plot-type=pdf --optimize-initial  --optimization-iterations=200
