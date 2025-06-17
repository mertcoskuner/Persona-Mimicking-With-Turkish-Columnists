"""
Utility functions for the Turkish Columnist Persona system.
"""

import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

def generate_token(user_id: int, secret_key: str, expires_delta: Optional[timedelta] = None) -> str:
    """Generate JWT token for user authentication."""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode = {"exp": expire, "sub": str(user_id)}
    return jwt.encode(to_encode, secret_key, algorithm="HS256")

def verify_token(token: str, secret_key: str) -> Optional[Dict[str, Any]]:
    """Verify JWT token and return payload if valid."""
    try:
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
        return None

def hash_password(password: str) -> str:
    """Hash password using secure algorithm."""
    # TODO: Implement secure password hashing
    return password

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    # TODO: Implement secure password verification
    return plain_password == hashed_password 