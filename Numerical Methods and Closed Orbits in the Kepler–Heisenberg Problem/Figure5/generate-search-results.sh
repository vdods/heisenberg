#!/bin/bash -e

# This dir needs to exactly match the one in generate.sh
BASE_DIR="Numerical Methods and Closed Orbits in the Kepler–Heisenberg Problem"
FIGURE_DIR="$BASE_DIR/Figure5"
OUTPUT_DIR="$FIGURE_DIR/generated-data/search-results"

if [ ! -d "$BASE_DIR" ]; then
    echo "This script must be run from the heisenberg project root (i.e. the one containing the files .gitignore and LICENSE.md)."
    exit -1
fi

mkdir -p "$OUTPUT_DIR"

# Run a search whose "abortive" subdirectory will contain some interesting quasi-periodic orbits,
# exiting after 11 tries (so the program is deterministic and doesn't rely on the user to
# abort after the desired results are generated).  Also note the --optimization-iterations=0 option,
# indicating that acceptable search candidates will not be optimized (because the point of this
# command is the plots that appear in the abortive subdirectory).
/usr/bin/time --verbose python3 -m heisenberg.search --dt=1.0e-2 --max-time=600 --seed=459540 --embedding-dimension=2 --embedding-solution-sheet-index=1 --plot-type=pdf --output-dir="$OUTPUT_DIR" --optimization-iterations=0 --exit-after-number-of-tries=11 --quantities-to-plot="x,y;t,z" --plot-size=3 --use-terse-plot-titles 2>&1 | tee "$OUTPUT_DIR/log.txt"
