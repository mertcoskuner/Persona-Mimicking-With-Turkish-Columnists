#!/bin/bash

# RAG Model Runner Script
# This script runs the RAG model with specified parameters

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0  # GPU device to use

# Configuration
MODEL_ID="ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1"
EMBEDDING_MODEL="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
QDRANT_URL="http://localhost:6333"
COLLECTION_NAME="abdulkadir_selvi_collection"

# Create results directory if it doesn't exist
mkdir -p results/rag

# Run RAG inference
echo "Starting RAG model inference..."
python src/rag/inference.py \
    --model_id "$MODEL_ID" \
    --embedding_model "$EMBEDDING_MODEL" \
    --qdrant_url "$QDRANT_URL" \
    --collection_name "$COLLECTION_NAME" \
    --output_dir "results/rag" \
    --max_new_tokens 512 \
    --temperature 0.6 \
    --top_p 0.9

# Run RAG evaluation if needed
if [ "$1" == "--evaluate" ]; then
    echo "Running RAG evaluation..."
    python src/rag/rag_engine.py \
        --predicted_file "data/predictions.jsonl" \
        --reference_file "data/references.json" \
        --output_scores_file "results/rag/scores.json"
fi

echo "RAG model execution completed!" 