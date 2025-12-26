#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "Building caliby with exponential backoff fix..."
cmake --build build --parallel 8

echo "Build complete!"
