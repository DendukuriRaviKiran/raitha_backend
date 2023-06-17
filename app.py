from flask import Flask, request, jsonify
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel
from flask_cors import CORS
import pinecone
import base64
import os
import requests
import openai
import json
import os
import yaml
import re

from langchain.agents import (
    create_json_agent,
    AgentExecutor
)
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.chains import LLMChain
from langchain.llms.openai import OpenAI
from langchain.requests import TextRequestsWrapper
from langchain.tools.json.tool import JsonSpec

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the CLIP model
model_ID = "openai/clip-vit-base-patch16"
model = CLIPModel.from_pretrained(model_ID)
preprocess = CLIPImageProcessor.from_pretrained(model_ID)

# Set up OpenAI API
OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY

# Define a function to load and preprocess the image
def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    image = preprocess(image, return_tensors="pt")
    return image

def extract_unique_ids_from_string(text):
    pattern = r'vec\d+'  # Regular expression pattern to match "vec" followed by digits
    
    ids = re.findall(pattern, text)  # Find all matches of the pattern in the text
    unique_ids = list(set(ids))  # Convert the list to a set to remove duplicates, then convert back to a list
    
    return unique_ids


@app.route('/generate_image', methods=['POST'])
def generate_image():
    if 'text' not in request.json:
        return jsonify({'error': 'No text found.'}), 400

    text = request.json['text']

    engine_id = "stable-diffusion-v1-5"
    api_host = os.getenv('API_HOST', 'https://api.stability.ai')
    api_key = "sk-RAxe3wMyc0UQ2cjZSdN3iJy58HA4d0pwYvnhZaD2Dvc9opM9"

    if api_key is None:
        raise Exception("Missing Stability API key.")

    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/text-to-image",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        json={
            "text_prompts": [
                {
                    "text": text
                }
            ],
            "cfg_scale": 7,
            "clip_guidance_preset": "FAST_BLUE",
            "height": 512,
            "width": 512,
            "samples": 1,
            "steps": 30,
        },
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()

    image_data = data["artifacts"][0]["base64"]
    image_path = 'generated_image.png'
    with open(image_path, "wb") as f:
        f.write(base64.b64decode(image_data))

    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode('utf-8')

    return jsonify({'image_data': encoded_image})

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found.'}), 400

    # Save the image file
    image_file = request.files['image']
    image_path = 'query_image.jpg'
    image_file.save(image_path)
    pinecone.init(api_key="3a4c9778-f1f0-4943-8a2f-297237a7cdda", environment="us-west1-gcp-free")
    INDEX_NAME = "my-index"
    index = pinecone.Index(INDEX_NAME)

    # Preprocess the image
    query_preprocess = load_and_preprocess_image(image_path)["pixel_values"]

    # Get the image embedding
    query_embedding = model.get_image_features(query_preprocess).tolist()[0]

    # Perform the search using Pinecone
    response = index.query(vector=query_embedding, top_k=5, include_metadata=True)

    response_dict = response.to_dict()

    response_dict.pop('namespace', None)
    
    # Return response as JSON
    return json.dumps(response_dict)


@app.route('/filter_data', methods=['GET'])
def filter_data():

    response_text = request.args.get('responseText')
    filter_text = request.args.get('filterText')

    
    response_text = json.loads(response_text) 
    data_dict = {}
    for item in response_text["matches"]:
        key = item['id']
        data_dict[key] = item["metadata"]
    
    messages = [ {"role": "system", "content": 
              "JSON Filtering Agent for  "} ]
    messages.append(
            {"role": "user", "content": str(data_dict) + " From the above dictionary, find the vector ID that satisfy the condition:  " + filter_text + " there can be only 2 possible outcomes if not ID's match the condition strictly just give the string False if not strictly give the ID strings that match the condition and if multiple exist seperate them with comma"},
        )
    chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613", messages=messages
        )
    reply = chat.choices[0].message.content

    ids = extract_unique_ids_from_string(reply)

    return ids

if __name__ == '__main__':
    app.run()
