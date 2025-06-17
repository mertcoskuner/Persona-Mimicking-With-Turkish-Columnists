"""
Configuration module for the project.
Contains all configurable parameters and settings.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# Model configurations
MODEL_CONFIGS = {
    "meta-llama": {
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "max_length": 2048,
        "temperature": 0.7,
        "top_p": 0.9,
    },
    "turkcell-llm": {
        "model_name": "turkcell/turkcell-llm-7b",
        "max_length": 2048,
        "temperature": 0.7,
        "top_p": 0.9,
    }
}

# Agent configurations
AGENT_CONFIGS = {
    "default": {
        "max_turns": 10,
        "context_window": 5,
        "persona_strength": 0.8,
    }
}

# RAG configurations
RAG_CONFIGS = {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "top_k": 3,
}

# API configurations
API_CONFIGS = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": False,
}

# Logging configurations
LOG_CONFIGS = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": PROJECT_ROOT / "logs" / "app.log",
}

# Create necessary directories
for directory in [DATA_DIR, RESULTS_DIR, MODELS_DIR, LOG_CONFIGS["file"].parent]:
    directory.mkdir(parents=True, exist_ok=True) 