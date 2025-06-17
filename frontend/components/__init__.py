"""
Frontend components for the Turkish Columnist Persona system.
"""

import streamlit as st
from typing import List, Dict, Any

def login_form() -> Dict[str, str]:
    """Render login form and return credentials."""
    st.subheader("Giriş Yap")
    username = st.text_input("Kullanıcı Adı")
    password = st.text_input("Şifre", type="password")
    return {"username": username, "password": password}

def persona_selector(personas: List[Dict[str, Any]]) -> int:
    """Render persona selection form and return selected persona ID."""
    st.subheader("Köşe Yazarı Seçin")
    persona_names = [p["name"] for p in personas]
    selected_name = st.selectbox("Köşe Yazarı", persona_names)
    return next(p["id"] for p in personas if p["name"] == selected_name)

def chat_interface(messages: List[Dict[str, Any]]) -> str:
    """Render chat interface and return new message."""
    st.subheader("Sohbet")
    
    # Display messages
    for msg in messages:
        if msg["role"] == "user":
            st.write(f"👤 Siz: {msg['content']}")
        else:
            st.write(f"🤖 {msg['content']}")
    
    # Input for new message
    new_message = st.text_input("Mesajınızı yazın...")
    return new_message 