"""
Configuration settings for the Turkish Columnist Persona system.
"""

import os
from typing import List

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Database Configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "persona_db")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")

# JWT Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRATION = int(os.getenv("JWT_EXPIRATION", "3600"))

# CORS Configuration
CORS_ORIGINS: List[str] = [
    "http://localhost:8501",  # Frontend development
    "http://localhost:3000",  # Alternative frontend
]

# Model Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "models")
BASE_MODEL = os.getenv("BASE_MODEL", "turkish-llama-8b")
DEVICE = os.getenv("DEVICE", "cuda")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")

# Rate Limiting
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "100"))  # requests per minute 