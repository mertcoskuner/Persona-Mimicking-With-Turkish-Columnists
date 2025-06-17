"""
Streamlit frontend application for the Turkish Columnist Persona project.
"""

import streamlit as st
import requests
from typing import Optional, Dict, List
import json
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"
PAGE_CONFIG = {
    "page_title": "Turkish Columnist Persona Chat",
    "page_icon": "🗣️",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "token" not in st.session_state:
        st.session_state.token = None
    if "selected_persona" not in st.session_state:
        st.session_state.selected_persona = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

# API functions
def login(username: str, password: str) -> bool:
    """Login to the API and get access token."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/token",
            data={"username": username, "password": password},
            timeout=10
        )
        if response.status_code == 200:
            tokens = response.json()
            st.session_state.token = tokens["access_token"]
            st.session_state.logged_in = True
            return True
        return False
    except Exception as e:
        st.error(f"Login error: {str(e)}")
        return False

def get_personas() -> List[str]:
    """Get list of available personas."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/personas",
            headers={"Authorization": f"Bearer {st.session_state.token}"}
        )
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"Error fetching personas: {str(e)}")
        return []

def send_message(persona: str, message: str) -> Optional[Dict]:
    """Send message to a persona and get response."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat/{persona}",
            json={"message": message},
            headers={"Authorization": f"Bearer {st.session_state.token}"}
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error sending message: {str(e)}")
        return None

# UI Components
def login_page():
    """Render login page."""
    st.title("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if username.strip() and password.strip():
                if login(username, password):
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Login failed. Please check your credentials.")
            else:
                st.warning("Please fill in both username and password.")

def chat_page():
    """Render chat page."""
    st.title("Chat with Turkish Columnists")
    
    # Sidebar for persona selection
    with st.sidebar:
        st.header("Select Persona")
        personas = get_personas()
        selected_persona = st.selectbox(
            "Choose a columnist",
            personas,
            index=0 if personas else None
        )
        
        if selected_persona:
            st.session_state.selected_persona = selected_persona
        
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.token = None
            st.rerun()
    
    # Main chat area
    if st.session_state.selected_persona:
        st.subheader(f"Chatting with {st.session_state.selected_persona}")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message to chat
            st.session_state.chat_history.append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().isoformat()
            })
            
            # Get response from persona
            response = send_message(st.session_state.selected_persona, prompt)
            if response:
                # Add persona response to chat
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response["message"],
                    "timestamp": datetime.now().isoformat()
                })
                st.rerun()

def main():
    """Main application entry point."""
    # Set page config
    st.set_page_config(**PAGE_CONFIG)
    
    # Initialize session state
    init_session_state()
    
    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .user-message {
            background-color: #e3f2fd;
        }
        .assistant-message {
            background-color: #f5f5f5;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Render appropriate page
    if not st.session_state.logged_in:
        login_page()
    else:
        chat_page()

if __name__ == "__main__":
    main()
