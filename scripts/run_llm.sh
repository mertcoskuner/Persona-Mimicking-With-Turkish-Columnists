#!/bin/bash

# LLM Model Runner Script
# This script runs the LLM model with specified parameters

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0  # GPU device to use

# Configuration
MODEL_ID="ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1"
OUTPUT_DIR="results/llm"

# Create results directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run LLM inference
echo "Starting LLM model inference..."
python src/llm/inference.py \
    --model_id "$MODEL_ID" \
    --output_dir "$OUTPUT_DIR" \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --top_p 0.9

# Run LLM evaluation if needed
if [ "$1" == "--evaluate" ]; then
    echo "Running LLM evaluation..."
    python src/llm/evaluation.py \
        --predicted_file "data/predictions.jsonl" \
        --reference_file "data/references.json" \
        --output_scores_file "$OUTPUT_DIR/scores.json"
fi

echo "LLM model execution completed!" 