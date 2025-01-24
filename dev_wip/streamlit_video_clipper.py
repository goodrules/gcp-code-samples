
import streamlit as st
from st_files_connection import FilesConnection
from google.cloud import aiplatform
from google.cloud import storage
import vertexai.generative_models
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Tool,
    Part,
    Image,
    SafetySetting,
    HarmCategory,
    HarmBlockThreshold,
)
import tempfile
import os
import mimetypes
from typing import Iterator
import time
import json

# Use this if using GCP - Vertex
from google.oauth2 import service_account

PROJECT_ID = 'mg-ce-demos' # change to your GCP project ID
REGION = 'us-central1' # change to the appropriate region

# Initialize Vertex AI
vertexai.init(project = PROJECT_ID, location = REGION)

video_generation_config=GenerationConfig(
        response_mime_type="application/json",
    )

safety_config = [
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=HarmBlockThreshold.OFF,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=HarmBlockThreshold.OFF,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=HarmBlockThreshold.OFF,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=HarmBlockThreshold.OFF,
    ),
]

def list_videos(bucket="cloud-samples-data"):
    # Create a client
    storage_client = storage.Client()
    
    # Specify the bucket name
    bucket_name = bucket
    
    # Get a bucket object
    bucket = storage_client.bucket(bucket_name)
    
    # List files in a bucket
    blobs = bucket.list_blobs(prefix="generative-ai/video/")
    blob_list=[]
    for blob in blobs:
        blob_list.append(blob.name.split('/')[-1])

    return blob_list

def video_analysis_content(prompt, video_path):
    prompt_input = f"INPUT: {prompt}"
    video_prompt_template = """    
    Please analyze the following video and identify key moments that could serve as standalone clips for the provided INPUT. Provide:
    
    1. A concise description of what happens in the clip
    2. The timestamp where the clip begins (in MM:SS format)
    3. The context or category of the clip (e.g., tutorial segment, key insight, demonstration)
    4. The recommended duration for the clip (in seconds)
    
    Please format your response as a JSON object with the following structure:
    
    {
      "clips": [
        {
          "clip_id": "string",
          "description": "string",
          "start_time": "number (seconds)",
          "category": "string",
          "recommended_duration": "number (seconds)",
          "keywords": ["string"]
        }
      ]
    }
    
    Important Guidelines:
    - Each clip should be self-contained and meaningful on its own
    - Clips should not be longer than 2-3 minutes unless absolutely necessary
    - Include relevant keywords for each clip to aid in categorization
    - Avoid overlapping clips unless the content warrants multiple perspectives
    
    Sample Input:
    "Please analyze this tutorial video on advanced Python programming techniques"
    
    Sample Output:
    {
      "clips": [
        {
          "clip_id": "clip_001",
          "description": "Introduction to list comprehensions with practical examples",
          "start_time": "45",
          "category": "tutorial_segment",
          "recommended_duration": 120,
          "keywords": ["list comprehension", "python basics", "code examples"]
        }
      ]
    }
    
    """
    video_prompt = prompt_input + video_prompt_template
    video_file = Part.from_uri(
        uri=video_path,
        mime_type="video/mp4",
    )
    contents = [video_file, video_prompt]
    return contents

def main():
    st.title("Analyze Videos")

    # Add clear history button in the sidebar
    with st.sidebar:
        model_name = st.selectbox(
            "Select Gemini Model",
            ["gemini-2.0-flash-exp", "gemini-1.5-pro-002", "gemini-1.5-flash-002"],
            index=0
        )
        system_prompt = st.text_area("System Prompt (optional)")
        if st.button("Reset"):
            st.cache_data.clear()
            #st.rerun()

    gemini2_flash = vertexai.generative_models.GenerativeModel(model_name) # new experimental model

    # Check if system prompt is empty
    if len(system_prompt) > 0:
        MODEL = vertexai.generative_models.GenerativeModel(model_name, system_instruction=system_prompt)
    else:
        MODEL = vertexai.generative_models.GenerativeModel(model_name)

    video_option = st.selectbox(
        "Which video do you want to analyze?",
        (list_videos()),
    )

    video_uri = "gs://cloud-samples-data/generative-ai/video/"+video_option
    video_url = "https://storage.googleapis.com/cloud-samples-data/generative-ai/video/"+video_option
    
    if len(video_option)>0:
        with st.expander("You selected: " + video_option):
            st.video(video_url)

    prompt_input = st.text_input("User input: ", value="What are the most interesting moments in this video?")
    submit = st.button("Analyze!")

    # Display the selected video
    #video_urls = []

    #If ask button is clicked
    if submit:
        contents = video_analysis_content(prompt_input, video_uri)
        for retry in range(10):
            while True:
                try:
                    response = gemini2_flash.generate_content(contents, generation_config=video_generation_config)
                    response_json = json.loads(response.text)
                except Exception as e:
                    print(f"Attempt {retry + 1} failed: {e}")
                    time.sleep(0.5 * 2 ** retry)
                break
        #clip_links = []
        #st.write(response_json)
        for clip in response_json["clips"]:
            start_time = int(clip["start_time"].split(":")[0])*60 + int(clip["start_time"].split(":")[1])
            end_time = int(start_time) + clip["recommended_duration"]
            #clip_links.append(link)
            #video_urls.append(video_url + f"#t={start_time},{end_time}")
            with st.expander(clip["description"] + ", starting at " + clip["start_time"] + ", " + str(clip["recommended_duration"]) + "s"):
                st.video(video_url, start_time=start_time, end_time=end_time)
                st.write(clip["keywords"])
                
if __name__ == "__main__":
    main()
