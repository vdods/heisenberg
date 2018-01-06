#!/bin/bash -e

BASE_DIR="NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem"
GENERATED_DATA_DIR="$BASE_DIR/generated-data" # Just for holding the generated command-list.txt and the overall log.txt
GENERATED_COMMAND_LIST="$GENERATED_DATA_DIR/command-list.txt"

if [ ! -d "$BASE_DIR" ]; then
    echo "This script must be run from the heisenberg project root (i.e. the one containing the files .gitignore and LICENSE.md)."
    exit -1
fi

echo "Generating all results.  Note that this will take a LONG time..."
echo "Generated files will appear in the generated-results subdirectories of their respective Figure/Table directories."

# Because we know we're running all computations, we can combine the single-threaded computations into
# a single command list to pipe to the `parallel` command.  This will maximize parallelism and minimize
# wall clock time.  Other computations are already internally parallelized (and/or have requirements about
# the order in which the scripts must be run), so they'll be run sequentially afterwards.

# Parallelizable
#     (Figures/Tables with command-list.txt)
#         Figure1
#         Figure3
#         Figure5 (quasi-periodic)
#         Figure7
#         Table1 (two command lists)
#     single-threaded computations
#         Figure2
#         Figure5 search results
#
# Non-parallelizable (already parallelized)
#     Figure4
#     Figure6 (2 sequential commands)

rm -rf "$GENERATED_DATA_DIR"
mkdir -p "$GENERATED_DATA_DIR"

cat "$BASE_DIR/Figure1/command-list.txt" >> "$GENERATED_COMMAND_LIST"
echo "\"$BASE_DIR/Figure2/generate.sh\"" >> "$GENERATED_COMMAND_LIST"
cat "$BASE_DIR/Figure3/command-list.txt" >> "$GENERATED_COMMAND_LIST"
cat "$BASE_DIR/Figure5/command-list.txt" >> "$GENERATED_COMMAND_LIST"
echo "\"$BASE_DIR/Figure5/generate-search-results.sh\"" >> "$GENERATED_COMMAND_LIST"
cat "$BASE_DIR/Figure7/command-list.txt" >> "$GENERATED_COMMAND_LIST"
cat "$BASE_DIR/Table1/command-list-for-orbits-up-to-order-6.txt" >> "$GENERATED_COMMAND_LIST"
cat "$BASE_DIR/Table1/command-list-for-orbits-of-order-7-to-8.txt" >> "$GENERATED_COMMAND_LIST"

# Note that because we're running some command lists for Figures/Tables directly, their generated-data directories
# may not have a log.txt file (because their own generate.sh file is what creates that, but we're skipping those).

/usr/bin/time bash -c "(parallel < \"$GENERATED_COMMAND_LIST\"; \"$BASE_DIR/Figure4/generate.sh\"; \"$BASE_DIR/Figure6/generate.sh\"; \"$BASE_DIR/Figure6/generate-local-minimum-plots.sh\")" 2>&1 | tee "$GENERATED_DATA_DIR/log.txt"
