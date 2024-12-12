
import streamlit as st
from google.cloud import aiplatform
import vertexai.generative_models
from vertexai.generative_models import GenerativeModel, Part, Content
import tempfile
import os
import mimetypes
from typing import Iterator
import time

PROJECT_ID = 'mg-ce-demos' # change to your GCP project ID
REGION = 'us-central1' # change to the appropriate region

# Initialize Vertex AI
vertexai.init(project = PROJECT_ID, location = REGION)

def process_uploaded_file(uploaded_file) -> Part:
    """Process uploaded file and convert to Gemini Part object."""
    if uploaded_file is None:
        return None
        
    # Create a temporary file to store the uploaded content
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Determine mime type
    mime_type, _ = mimetypes.guess_type(uploaded_file.name)
    if mime_type is None:
        mime_type = "application/octet-stream"

    # Create Part object from file
    with open(tmp_file_path, "rb") as f:
        file_content = f.read()
        
    # Clean up temporary file
    os.unlink(tmp_file_path)
    
    return Part.from_data(data=file_content, mime_type=mime_type)

def stream_chat(
    chat_history: list, 
    query: str, 
    uploaded_part: Part = None, 
    model=vertexai.generative_models.GenerativeModel("gemini-2.0-flash-exp")
) -> Iterator[str]:
    """Stream chat responses from Gemini."""
    # Initialize chat
    formatted_history = []
    for msg in chat_history:
        formatted_history.extend([
            Content(role="user", parts=[Part.from_text(str(msg["user"]))]),
            Content(role="model", parts=[Part.from_text(str(msg["model"]))])
        ])
    
    chat = model.start_chat(history=formatted_history)
    
    # Prepare message parts
    message_parts = [query]
    if uploaded_part:
        message_parts.insert(0, uploaded_part)
    
    # Get streaming response
    response = chat.send_message(message_parts, stream=True)
    
    for chunk in response:
        if hasattr(chunk, "text"):
            yield chunk.text

def main():
    st.title("Chat with Gemini")

    # Add clear history button in the sidebar
    with st.sidebar:
        model_name = st.selectbox(
            "Select Gemini Model",
            ["gemini-2.0-flash-exp", "gemini-1.5-pro-002", "gemini-1.5-flash-002"],
            index=0
        )
        system_prompt = st.text_area("System Prompt (optional)")
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    # Check if system prompt is empty
    if len(system_prompt) > 0:
        MODEL = vertexai.generative_models.GenerativeModel(model_name, system_instruction=system_prompt)
    else:
        MODEL = vertexai.generative_models.GenerativeModel(model_name)
    
    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a file (optional)", type=["txt", "pdf", "png", "jpg", "jpeg"])
    
    # Convert uploaded file to Part if present
    uploaded_part = process_uploaded_file(uploaded_file) if uploaded_file else None
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(message["user"])
        with st.chat_message("model"):
            st.write(message["model"])
    
    # Handle new user input
    if user_input:
        # Show user message
        with st.chat_message("user"):
            st.write(user_input)
            
        # Show assistant response with streaming
        with st.chat_message("model"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Stream the response
            for chunk in stream_chat(st.session_state.chat_history, user_input, uploaded_part, model=MODEL):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.05)
            
            # Update final response
            message_placeholder.markdown(full_response)
        
        # Update chat history
        st.session_state.chat_history.append({
            "user": user_input,
            "model": full_response
        })
        
if __name__ == "__main__":
    main()
