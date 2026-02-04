#!/bin/bash
set -e

echo "========================================="
echo "Triton-Seq Environment Setup"
echo "========================================="
echo ""

# Check CUDA
echo "Checking CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: CUDA not found. Please install CUDA 12.1 or later."
    echo "Visit: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo "✓ Found CUDA version: $CUDA_VERSION"
echo ""

# Check conda
echo "Checking conda installation..."
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Miniconda or Anaconda."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo "✓ Found conda"
echo ""

# Create conda environment
echo "Creating conda environment 'triton-seq'..."
if conda env list | grep -q "^triton-seq "; then
    echo "Environment 'triton-seq' already exists."
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n triton-seq -y
        conda env create -f environment.yml
    fi
else
    conda env create -f environment.yml
fi
echo ""

# Activate environment
echo "Environment created. To activate it, run:"
echo "  conda activate triton-seq"
echo ""

# Ask about custom Triton
read -p "Do you want to install the custom Triton compiler with shared memory optimizations? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "To install the custom Triton compiler, run:"
    echo "  conda activate triton-seq"
    echo "  bash scripts/build_triton.sh"
else
    echo "You can install the custom compiler later by running:"
    echo "  bash scripts/build_triton.sh"
fi
echo ""

echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. conda activate triton-seq"
echo "  2. python examples/basic_alignment.py"
echo "  3. See docs/QUICKSTART.md for more information"
echo ""
