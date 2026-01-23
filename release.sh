#!/bin/bash
# Release script for publishing caliby to PyPI
# Usage: ./release.sh

set -e  # Exit on any error

echo "🚀 Starting caliby release process..."
echo ""

# Get current version from pyproject.toml
VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
echo "📦 Version: $VERSION"
echo ""

# Confirm before proceeding
read -p "Continue with release v$VERSION? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Release cancelled"
    exit 1
fi

# Step 1: Clean old build artifacts
echo "🧹 Cleaning old build artifacts..."
rm -rf build/ dist/ *.egg-info
echo "✓ Cleaned"
echo ""

# Step 2: Build distributions
echo "🔨 Building distributions..."
python3 -m build
echo "✓ Built successfully"
echo ""

# Step 3: Check distributions with twine
echo "🔍 Checking distributions..."
python3 -m twine check dist/*
echo "✓ All checks passed"
echo ""

# Step 4: Show what will be uploaded
echo "📤 Will upload:"
ls -lh dist/
echo ""

# Step 5: Upload to PyPI (source distribution only)
echo "⬆️  Uploading to PyPI..."
python3 -m twine upload dist/*.tar.gz
echo ""

echo "✅ Release v$VERSION completed successfully!"
echo "🌐 View at: https://pypi.org/project/caliby/$VERSION/"
echo ""
echo "📝 Don't forget to:"
echo "   - Create git tag: git tag v$VERSION && git push origin v$VERSION"
echo "   - Update CHANGELOG.md"
echo "   - Create GitHub release"
