import streamlit as st
import anthropic
from anthropic import AnthropicVertex

PROJECT_ID = 'mg-ce-demos'
REGION = 'us-east5'

# Anthropic Vertex Client
anthropic_client = AnthropicVertex(region=REGION, project_id=PROJECT_ID)

# Anthropic Model
anthr_claude_sonnet35_v2 = "claude-3-5-sonnet-v2"
anthr_claude_opus = "claude-3-opus"
anthr_claude_sonnet37 = "claude-3-7-sonnet"  # Adding Claude 3.7 Sonnet

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to stream response from Claude
def stream_claude_response(model=anthr_claude_sonnet37):
    try:
        # Set stream=True to get streaming response
        stream_response = anthropic_client.messages.create(
            model=model,
            max_tokens=4000,
            temperature=0.5,
            system="",
            messages=st.session_state.messages,
            stream=True  # Enable streaming
        )
        
        # Return the stream object
        return stream_response
    except Exception as e:
        # Return the error as a string
        return f"An error occurred: {str(e)}"

# Streamlit app
st.markdown("<h1 style='color:firebrick'>Claude on Vertex Chatbot</h1>", unsafe_allow_html=True)

# Add clear history button in the sidebar
with st.sidebar:
    st.markdown("# <div style='color:firebrick'>About</div>", unsafe_allow_html=True)
    st.markdown("### A simple chatbot using the Anthropic API to interact with Claude.")
    st.markdown("""<div style='color:darkslategrey;font-size:smaller'>
    Streamlit app using Claude 3.5 Sonnet, Claude 3.7 Sonnet, and Claude 3 Opus.""",
    unsafe_allow_html=True
    )
    model_name = st.selectbox(
        "Select Claude Model",
        [anthr_claude_sonnet37, anthr_claude_sonnet35_v2, anthr_claude_opus],
        index=0
    )
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to ask Claude?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get Claude's response with streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Get streaming response
        stream = stream_claude_response(model_name)
        
        # If an error occurred (stream is a string), display it
        if isinstance(stream, str):
            message_placeholder.write(stream)
            full_response = stream
        else:
            # Process the streaming response
            for chunk in stream:
                # Check if the chunk has content
                if chunk.type == "content_block_delta" and hasattr(chunk.delta, "text"):
                    # Extract the text from the chunk
                    text_chunk = chunk.delta.text
                    # Add the chunk to the full response
                    full_response += text_chunk
                    # Update the displayed message with the accumulated response
                    message_placeholder.markdown(full_response)
    
    # Add Claude's response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
