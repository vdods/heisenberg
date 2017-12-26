#!/bin/bash -e

# This dir needs to exactly match the one in generate-orbits-up-to-order-6.sh
BASE_DIR="Numerical Methods and Closed Orbits in the Kepler–Heisenberg Problem"
TABLE_DIR="$BASE_DIR/Table1"
OUTPUT_DIR="$TABLE_DIR/generated-data"

if [ ! -d "$BASE_DIR" ]; then
    echo "This script must be run from the heisenberg project root (i.e. the one containing the files .gitignore and LICENSE.md)."
    exit -1
fi

mkdir -p "$OUTPUT_DIR"

# The 1/1 orbit -- in the paper, this one was found via heisenberg.search and computed separately, so --optimize-initial
# is not used here.  However, adding the options
#     --optimize-initial --optimization-iterations=200
# will produce a refined result (though that gives a result that's different than the paper).
python3 -m heisenberg.plot --dt=0.003 --max-time=20.00000000000000 --initial-preimage=[3.043097949152258e-06] --embedding-dimension=1 --embedding-solution-sheet-index=0 --output-dir="$OUTPUT_DIR" --quantities-to-plot="x,y;t,z;error(H);error(J);sqd;class-signal;objective" --plot-type=pdf
