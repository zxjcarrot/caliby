#!/bin/bash
# Run Caliby tests
#
# Usage:
#   ./run_tests.sh           # Run all tests
#   ./run_tests.sh -k hnsw   # Run only HNSW tests
#   ./run_tests.sh -v        # Run with verbose output

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if caliby module is installed (via pip install -e .)
if ! python3 -c "import caliby" 2>/dev/null; then
    echo "Error: caliby module not installed. Please build the project first:"
    echo "  ./rebuild.sh"
    echo "  OR"
    echo "  pip install -e ."
    exit 1
fi

# Create temporary directory for tests
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Create heapfile in temp directory
touch "$TEMP_DIR/heapfile"

# Run tests from temp directory
cd "$TEMP_DIR"

# Run pytest (capture exit code to handle cleanup properly)
python3 -m pytest "$SCRIPT_DIR/tests" "$@"
EXIT_CODE=$?

# Exit with pytest's exit code
exit $EXIT_CODE
