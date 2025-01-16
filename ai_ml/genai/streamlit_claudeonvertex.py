
import streamlit as st
import anthropic
from anthropic import AnthropicVertex

PROJECT_ID = 'mg-ce-demos'
REGION = 'us-east5'

# Anthropic Vertex Client
anthropic_client = AnthropicVertex(region=REGION, project_id=PROJECT_ID)

# Anthropic Model
anthr_claude_sonnet35_v2 = "claude-3-5-sonnet-v2@20241022"
anthr_claude_opus = "claude-3-opus"

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to get response from Claude
def get_claude_response(prompt, model=anthr_claude_sonnet35_v2, ):
    try:
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=4000,
            temperature=0.5,
            system = "",
            messages = st.session_state.messages
        )
        return response.content[0].text
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit app
st.markdown("<h1 style='color:firebrick'>Claude on Vertex Chatbot</h1>",unsafe_allow_html=True)

# Add clear history button in the sidebar
with st.sidebar:
    st.markdown("# <div style='color:firebrick'>About</div>",unsafe_allow_html=True)
    st.markdown("### A simple chatbot using the Anthropic API to interact with Claude.")
    st.markdown("""<div style='color:darkslategrey;font-size:smaller'>
    Streamlit app using version 3.5 of Claude Sonnet and version 3 of Claude Opus.""",
    unsafe_allow_html=True
    )
    model_name = st.selectbox(
        "Select Claude Model",
        [anthr_claude_sonnet35_v2, anthr_claude_opus],
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

    # Get Claude's response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = get_claude_response(prompt, model_name)
        message_placeholder.write(full_response)
    
    # Add Claude's response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
