#!/bin/bash -e

BASE_DIR="NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem"
FIGURE_DIR="$BASE_DIR/Figure2"
OUTPUT_DIR="$FIGURE_DIR/generated-data"

if [ ! -d "$BASE_DIR" ]; then
    echo "This script must be run from the heisenberg project root (i.e. the one containing the files .gitignore and LICENSE.md)."
    exit -1
fi

mkdir -p "$OUTPUT_DIR"

echo "Generating 'Figure 2' plots..."
echo "Generated files will appear in the directory \"$OUTPUT_DIR\""
/usr/bin/time --verbose python3 -m heisenberg.search --dt=1.0e-2 --max-time=200 --seed=123456789 --embedding-dimension=2 --embedding-solution-sheet-index=1 --plot-type=pdf --output-dir="$OUTPUT_DIR" --exit-after-number-of-successes=1 --quantities-to-plot="x,y;t,z;sqd;objective" --use-terse-plot-titles --plot-size=3 2>&1 | tee "$OUTPUT_DIR/log.txt"
