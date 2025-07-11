#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Deactivate conda if it's active
if [ -n "$CONDA_PREFIX" ]; then
    echo "Deactivating conda environment..."
    conda deactivate
fi

# Unset conda-related environment variables to avoid conflicts
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset CONDA_PROMPT_MODIFIER
unset CONDA_SHLVL
unset CONDA_PYTHON_EXE
unset CONDA_EXE

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    /usr/bin/python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install pandas numpy torch pyarrow
else
    source .venv/bin/activate
fi

# Set environment variables for PyO3
export PYTHONHOME="$SCRIPT_DIR/.venv"
export VIRTUAL_ENV="$SCRIPT_DIR/.venv"
export PATH="$SCRIPT_DIR/.venv/bin:$PATH"

# Set the Python path to include standard library
export PYTHONPATH="$SCRIPT_DIR/.venv/lib/python3.13:$SCRIPT_DIR/.venv/lib/python3.13/lib-dynload:$SCRIPT_DIR/.venv/lib/python3.13/site-packages"

# For macOS, set the dynamic library path
if [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH="$SCRIPT_DIR/.venv/lib:$DYLD_LIBRARY_PATH"
fi

# Explicitly tell PyO3 which Python to use
export PYO3_PYTHON="$SCRIPT_DIR/.venv/bin/python"

# Run the Rust program
echo "Using Python from: $(which python)"
echo "Python version: $(python --version)"
echo "Python home: $PYTHONHOME"
echo "Running Rust program..."

cargo run --release --bin read_bruker_data