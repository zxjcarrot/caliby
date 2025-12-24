#!/bin/bash
# Run Caliby tests
#
# Usage:
#   ./run_tests.sh           # Run all tests
#   ./run_tests.sh -k hnsw   # Run only HNSW tests
#   ./run_tests.sh -v        # Run with verbose output

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if build exists
if [ ! -d "build" ]; then
    echo "Error: build directory not found. Please build the project first:"
    echo "  mkdir build && cd build && cmake .. && make -j\$(nproc)"
    exit 1
fi

# Check if the Python module exists
if ! ls build/caliby*.so 1> /dev/null 2>&1; then
    echo "Error: caliby module not found. Please build the project first:"
    echo "  cd build && make -j\$(nproc)"
    exit 1
fi

# Create temporary directory for tests
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Create heapfile in temp directory
touch "$TEMP_DIR/heapfile"

# Run tests from temp directory
cd "$TEMP_DIR"

# Add build directory to PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR/build:$PYTHONPATH"

# Run pytest (capture exit code to handle cleanup properly)
python3 -m pytest "$SCRIPT_DIR/tests" "$@"
EXIT_CODE=$?

# Exit with pytest's exit code
exit $EXIT_CODE
