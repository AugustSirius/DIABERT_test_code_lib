#!/bin/bash

echo "Setting up conda environment 'siri' for the project..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda/Miniconda first."
    exit 1
fi

# Activate the siri environment
echo "Activating siri environment..."
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate siri

# Check Python version
echo "Python version in siri environment:"
python --version

# Install required packages
echo "Installing required packages in siri environment..."
pip install pandas numpy torch pyarrow

# Verify installation
echo -e "\nVerifying package installation..."
python -c "
import pandas
import numpy
import torch
import pyarrow
print('All packages imported successfully!')
print(f'pandas version: {pandas.__version__}')
print(f'numpy version: {numpy.__version__}')
print(f'torch version: {torch.__version__}')
print(f'pyarrow version: {pyarrow.__version__}')
"

echo -e "\nSetup complete!"