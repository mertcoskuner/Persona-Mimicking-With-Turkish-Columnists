#!/bin/bash

# Main Runner Script
# This script runs all models in sequence

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Python is installed
if ! command_exists python; then
    echo "Error: Python is not installed"
    exit 1
fi

# Check if required Python packages are installed
echo "Checking required packages..."
python -c "
import sys
required_packages = ['torch', 'transformers', 'sentence-transformers', 'qdrant-client']
missing_packages = []
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)
if missing_packages:
    print(f'Missing packages: {', '.join(missing_packages)}')
    sys.exit(1)
"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements if needed
if [ ! -f "venv/.requirements_installed" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
    touch venv/.requirements_installed
fi

# Function to run a script and check its status
run_script() {
    local script=$1
    local name=$2
    echo "Running $name..."
    if bash "$script"; then
        echo "$name completed successfully"
    else
        echo "Error: $name failed"
        exit 1
    fi
}

# Run all models
echo "Starting all models..."

# Run RAG model
run_script "scripts/run_rag.sh" "RAG model"

# Run LLM model
run_script "scripts/run_llm.sh" "LLM model"

# Run fine-tuning
run_script "scripts/run_finetune.sh" "Fine-tuning"

echo "All models completed successfully!"

# Deactivate virtual environment
deactivate 