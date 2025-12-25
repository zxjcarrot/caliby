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
echo "[0/4] Updating version number..."
TIMESTAMP=$(date +%Y%m%d%H%M%S)
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

# Step 2: Install using pip (uses setup.py configuration)
echo "[2/4] Installing caliby with pip (uses setup.py config)..."
pip install -e . --force-reinstall --no-build-isolation
echo "✓ Build and install complete"
echo ""

# Step 3: Update version in Python module
echo "[3/4] Updating Python module version..."
VERSION_LINE="m.attr(\"__version__\") = \"${NEW_VERSION}\";"
if grep -q "__version__" src/bindings.cpp; then
    sed -i "s|m\.attr(\"__version__\").*|${VERSION_LINE}|" src/bindings.cpp
else
    # Add version after module doc string
    sed -i "/m\.doc()/a\\    ${VERSION_LINE}" src/bindings.cpp
fi
# Rebuild to include version update
echo "  Rebuilding with version..."
pip install -e . --force-reinstall --no-build-isolation
echo "✓ Version embedded in module"
echo ""

# Step 4: Verify installation
echo "[4/4] Verifying installation..."
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
