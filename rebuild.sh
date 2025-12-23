#!/bin/bash
#
# Caliby Full Rebuild Script
# Cleans, rebuilds, and installs caliby from scratch
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Caliby Full Rebuild Script"
echo "========================================"
echo ""

# Step 0: Update version number with timestamp
echo "[0/7] Updating version number..."
TIMESTAMP=$(date +%Y%m%d.%H%M%S)
NEW_VERSION="0.1.0.dev${TIMESTAMP}"
sed -i "s/^version = .*/version = \"${NEW_VERSION}\"/" pyproject.toml
echo "✓ Version updated to: $NEW_VERSION"
echo ""

# Step 1: Clean up old build artifacts
echo "[1/7] Cleaning up old build artifacts..."
rm -rf build/
rm -rf caliby.egg-info/
rm -rf dist/
rm -f *.so
rm -f heapfile
rm -f catalog.dat
# Uninstall any system-installed versions
if pip3 show caliby > /dev/null 2>&1; then
    echo "  Uninstalling system-installed caliby..."
    pip3 uninstall -y caliby > /dev/null 2>&1 || sudo pip3 uninstall -y caliby > /dev/null 2>&1
fi
echo "✓ Cleanup complete"
echo ""

# Step 2: Create fresh build directory
echo "[2/7] Creating build directory..."
mkdir -p build
cd build
echo "✓ Build directory created"
echo ""

# Step 3: Run CMake configuration
echo "[3/7] Running CMake configuration..."
cmake .. \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_CXX_FLAGS="-g -fno-omit-frame-pointer -DCALICO_SPECIALIZATION_CALICO"
echo "✓ CMake configuration complete"
echo ""

# Step 4: Build with all CPU cores
echo "[4/7] Building caliby..."
make -j$(nproc)
echo "✓ Build complete"
echo ""

# Step 5: Copy module to root directory
echo "[5/7] Installing module..."
MODULE_FILE=$(ls caliby.cpython-*.so 2>/dev/null | head -1)
if [ -n "$MODULE_FILE" ]; then
    cp "$MODULE_FILE" ..
    echo "✓ Module copied to root: $MODULE_FILE"
else
    echo "✗ Error: Module file not found!"
    exit 1
fi
cd ..
echo ""

# Step 6: Update version in Python module
echo "[6/7] Updating Python module version..."
VERSION_LINE="m.attr(\"__version__\") = \"${NEW_VERSION}\";"
if grep -q "__version__" src/bindings.cpp; then
    sed -i "s|m\.attr(\"__version__\").*|${VERSION_LINE}|" src/bindings.cpp
else
    # Add version after module doc string
    sed -i "/m\.doc()/a\\    ${VERSION_LINE}" src/bindings.cpp
fi
# Rebuild to include version update
cd build
echo "  Rebuilding with version..."
make -j$(nproc)
MODULE_FILE=$(ls caliby.cpython-*.so 2>/dev/null | head -1)
if [ -n "$MODULE_FILE" ]; then
    cp "$MODULE_FILE" ..
    echo "  ✓ Module with version copied: $MODULE_FILE"
else
    echo "  ✗ Error: Module file not found after version rebuild!"
    exit 1
fi
cd ..
echo "✓ Version embedded in module"
echo ""

# Step 7: Verify installation
echo "[7/7] Verifying installation..."
if python3 -c "import caliby; print('  Version:', caliby.__version__ if hasattr(caliby, '__version__') else 'N/A')" 2>&1; then
    echo "✓ Caliby module loaded successfully"
else
    echo "✗ Error: Failed to load caliby module"
    exit 1
fi
echo ""

echo "========================================"
echo "Rebuild Complete!"
echo "========================================"
echo ""
echo "You can now run examples:"
echo "  cd examples && python3 benchmark.py"
echo "  cd examples && python3 hnsw_example.py"
echo "  cd benchmark && python3 compare_hnsw.py"
echo ""
