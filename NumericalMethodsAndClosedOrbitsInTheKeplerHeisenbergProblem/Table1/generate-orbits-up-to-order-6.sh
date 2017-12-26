#!/bin/bash -e

BASE_DIR="NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem"
TABLE_DIR="$BASE_DIR/Table1"
OUTPUT_DIR="$TABLE_DIR/generated-data"

if [ ! -d "$BASE_DIR" ]; then
    echo "This script must be run from the heisenberg project root (i.e. the one containing the files .gitignore and LICENSE.md)."
    exit -1
fi

mkdir -p "$OUTPUT_DIR"

echo "Generating 'Table 1' plots (thereby solving for the initial conditions of each orbit of symmetry order up to 6) in parallel..."
echo "Generated files will appear in the directory \"$OUTPUT_DIR\""
cat "$TABLE_DIR/command-list-for-orbits-up-to-order-6.txt" | /usr/bin/time --verbose parallel 2>&1 | tee "$OUTPUT_DIR/log-for-orbits-up-to-order-6.txt"
