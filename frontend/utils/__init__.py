"""
Utility functions for the frontend of Turkish Columnist Persona system.
"""

import requests
from typing import Dict, List, Any, Optional
import streamlit as st

API_BASE_URL = "http://localhost:8000/api/v1"

def login(username: str, password: str) -> Optional[str]:
    """Login and return JWT token."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/login",
            json={"username": username, "password": password}
        )
        response.raise_for_status()
        return response.json()["access_token"]
    except requests.RequestException:
        st.error("Giriş başarısız. Lütfen bilgilerinizi kontrol edin.")
        return None

def get_personas(token: str) -> List[Dict[str, Any]]:
    """Get list of available personas."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/personas",
            headers={"Authorization": f"Bearer {token}"}
        )
        response.raise_for_status()
        return response.json()["personas"]
    except requests.RequestException:
        st.error("Köşe yazarları yüklenemedi.")
        return []

def create_chat_session(token: str, persona_id: int) -> Optional[int]:
    """Create new chat session and return session ID."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat/sessions",
            headers={"Authorization": f"Bearer {token}"},
            json={"persona_id": persona_id}
        )
        response.raise_for_status()
        return response.json()["session_id"]
    except requests.RequestException:
        st.error("Sohbet başlatılamadı.")
        return None

def send_message(token: str, session_id: int, content: str) -> Optional[Dict[str, Any]]:
    """Send message and return response."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat/sessions/{session_id}/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"content": content}
        )
        response.raise_for_status()
        return response.json()["message"]
    except requests.RequestException:
        st.error("Mesaj gönderilemedi.")
        return None

def get_messages(token: str, session_id: int) -> List[Dict[str, Any]]:
    """Get chat history."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/chat/sessions/{session_id}/messages",
            headers={"Authorization": f"Bearer {token}"}
        )
        response.raise_for_status()
        return response.json()["messages"]
    except requests.RequestException:
        st.error("Mesaj geçmişi yüklenemedi.")
        return [] 