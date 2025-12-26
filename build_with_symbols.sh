#!/bin/bash
# Build caliby with debug symbols (not stripped) for profiling

set -e

cd "$(dirname "$0")"

echo "Building caliby with debug symbols for profiling..."

# Clean
rm -rf build
mkdir -p build
cd build

# Configure with debug symbols, no stripping
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -g -fno-omit-frame-pointer -march=native -DNDEBUG -DCALICO_SPECIALIZATION_CALICO" \
    -DCMAKE_STRIP=/bin/true \
    -DBUILD_SHARED_LIBS=ON \
    -DCALIBY_BUILD_PYTHON=ON

# Build
make -j$(nproc)

# Copy to root (unstripped version)
cp caliby.cpython-*.so ..

echo ""
echo "✓ Build complete with symbols!"
echo "Library: $(ls ../caliby.cpython-*.so)"
echo ""
echo "Verify symbols:"
nm -C ../caliby.cpython-*.so | grep -E "searchLayer|addPoint" | head -3
