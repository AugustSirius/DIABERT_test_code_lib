#!/bin/bash

echo "Setting up Python environment for Rust project..."

# Remove existing virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf .venv
fi

# Create new virtual environment
echo "Creating new virtual environment..."
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing Python packages..."
pip install pandas numpy torch pyarrow

# Save the exact versions for reproducibility
pip freeze > requirements-lock.txt

echo "Python environment setup complete!"
echo "Virtual environment location: $(pwd)/.venv"
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"

# Test imports
echo -e "\nTesting Python imports..."
python -c "import pandas; print('✓ pandas imported successfully')"
python -c "import numpy; print('✓ numpy imported successfully')"
python -c "import torch; print('✓ torch imported successfully')"
python -c "import pyarrow; print('✓ pyarrow imported successfully')"

echo -e "\nSetup complete! Use './run.sh' to run the project."