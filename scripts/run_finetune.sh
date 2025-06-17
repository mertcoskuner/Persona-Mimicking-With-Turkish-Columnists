#!/bin/bash

# Fine-tuning Runner Script
# This script runs the fine-tuning process with specified parameters

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0  # GPU device to use

# Configuration
BASE_MODEL="ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1"
TRAINING_DATA="data/training_data.jsonl"
OUTPUT_DIR="results/finetune"
NUM_EPOCHS=3
BATCH_SIZE=4
LEARNING_RATE=2e-5

# Create results directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run fine-tuning
echo "Starting fine-tuning process..."
python src/training/finetune.py \
    --base_model "$BASE_MODEL" \
    --training_data "$TRAINING_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --save_steps 100 \
    --eval_steps 100

# Run evaluation after fine-tuning if needed
if [ "$1" == "--evaluate" ]; then
    echo "Running fine-tuned model evaluation..."
    python src/training/evaluate.py \
        --model_path "$OUTPUT_DIR/final_model" \
        --test_data "data/test_data.jsonl" \
        --output_file "$OUTPUT_DIR/evaluation_results.json"
fi

echo "Fine-tuning process completed!" 