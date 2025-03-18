
import streamlit as st
from google import genai # new unified SDK
from google.genai import types
import tempfile
import os
import mimetypes
from typing import Iterator
import time
import json

PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = "us-central1"

# Use this if using GCP - Vertex
from google.oauth2 import service_account
import os

credentials = service_account.Credentials.from_service_account_file(
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'],
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)

google_genai_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION, credentials=credentials)

system_context_default = "You are a specialized AI assistant focused on creative content and marketing strategy. Always generate properly formatted JSON objects according to the specified structure.  Do not ask for feedback, only respond with content."

cluster_info = """cluster 1, average spend $49.44, count of orders per person 1.23, days since last order 102.87 \n
cluster 2, average spend $59.56, count of orders per person 3.51, days since last order 87.9 \n
cluster 3, average spend $251.34, count of orders per person 1.14, days since last order 205.85 \n
cluster 4, average spend $57.47, count of orders per person 3.49, days since last order 354.32 \n
cluster 5, average spend $48.87, count of orders per person 1.22, days since last order 376.5 \n
"""

def generate(
    query: str, 
    system_context=system_context_default,
    model="gemini-2.0-flash-001",
) -> Iterator[str]:
    """Stream chat responses from Gemini."""
    generate_content_config = types.GenerateContentConfig(
        temperature = 1,
        top_p = 0.95,
        max_output_tokens = 4096,
        response_modalities = ["TEXT"],
        response_mime_type="application/json",
        system_instruction = system_context,
        safety_settings = [types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="OFF"
        ),types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="OFF"
        ),types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="OFF"
        ),types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="OFF"
        )]
    )

    prompt = f"""{query}

    Cluster details:
    {cluster_info}

    Respond ONLY with a JSON object in this exact format:
        {{
            "personas": [
                {{
                    "title": "persona name",
                    "description": "persona",
                    "favorite": "favorite object, such as sunglasses",
                    "image_prompt": "prompt used for image generation",
                }},
            ]
        }}
    """
    
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)]
        )
    ]
    
    # Get full response
    response = google_genai_client.models.generate_content(model=model, contents=contents, config=generate_content_config)
    return response.text

def retry_imagen(prompt, model_name, retries = 3):
    for retry in range(retries):
        while True:
            try:
                response_image = google_genai_client.models.generate_images(
                    model=model_name,
                    prompt=prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        include_rai_reason=True,
                        #output_mime_type='image/jpeg',
                    )
                )
            except Exception as e:
                print(f"Attempt {retry + 1} failed: {e}")
                time.sleep(0.5 * 2 ** retry)
#                continue
            break
    return response_image

st.title("Gemini Marketing Bot")

# Add clear history button in the sidebar
with st.sidebar:
    model_name = st.selectbox(
        "Select Gemini Model",
        ["gemini-2.0-flash-001"],
        index=0
    )
    image_model_name = st.selectbox(
        "Select Imagen Model",
        ["imagen-3.0-generate-002", "imagen-3.0-fast-generate-001"],
        index=0
    )
    system_prompt = st.text_area("System Prompt (optional)")
    if st.button("Clear"):
        #st.session_state.chat_history = []
        st.session_state["user_input"] = ""
        st.rerun()


MODEL = model_name
IMAGE_MODEL = image_model_name

# Chat input
st.markdown(str("## Here are our clusters: \n\n" + cluster_info))
user_input = st.text_area("Let's analyze", value="Analyse the following clusters and come up with a creative brand persona for each that includes the detail of their favorite kind of sunglasses and a prompt to generate an image including an animal wearing their favorite type of sunglasses.")

persona_list = []
image_list = []

if st.button("Generate Text Response"):
    response = generate(user_input, system_prompt, model=MODEL)
    # Parse JSON data
    data = json.loads(response)
    
    # Ensure we have personas data
    if 'personas' not in data:
        print("Error: No personas found in JSON data")
    
    # Loop through each persona
    for i, persona in enumerate(data['personas'], 1):
        persona_str = f"**PERSONA {i}**: {persona.get('title', 'Untitled')} \n\n" + f"**Description**: {persona.get('description', 'No description available')} \n\n" + f"**Favorite Product**: {persona.get('favorite', 'No favorite specified')} \n\n" + f"**Image Prompt**: {persona.get('image_prompt', 'No image prompt available')} \n\n"
        persona_list.append(persona)
        response_image = retry_imagen(persona.get('image_prompt'), IMAGE_MODEL)
        image_list.append(response_image)
        st.markdown(persona_str)
        st.image(response_image.generated_images[0].image.image_bytes, caption=["Generated by Imagen 3"])
        
