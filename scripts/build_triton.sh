#!/bin/bash
set -e

echo "========================================="
echo "Building Custom Triton Compiler"
echo "========================================="
echo ""

# Check if we're in the right environment
if [[ "$CONDA_DEFAULT_ENV" != "triton-seq" ]]; then
    echo "ERROR: Please activate the triton-seq environment first:"
    echo "  conda activate triton-seq"
    exit 1
fi

# Check if submodule is initialized
if [ ! -d "compiler/triton/.git" ]; then
    echo "Initializing Triton compiler submodule..."
    git submodule update --init --recursive compiler/triton
fi

echo "Building Triton from source..."
echo "This may take 10-20 minutes..."
echo ""

cd compiler/triton

# Install in development mode.
# -DTRITON_BUILD_TESTING=OFF avoids network failures when fetching googletest,
# which is only needed for C++ unit tests, not for the Python library itself.
TRITON_BUILD_TESTING=OFF pip install -e python

cd ../..

echo ""
echo "========================================="
echo "Custom Triton Compiler Built Successfully!"
echo "========================================="
echo ""
echo "Testing installation..."
python -c "import triton; print(f'Triton version: {triton.__version__}')"
echo ""
echo "You can now run benchmarks with the optimized compiler:"
echo "  python benchmarks/scripts/run_baseline.py"
echo ""
