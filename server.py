# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
from fastapi.middleware.cors import CORSMiddleware
import random
import logging


# Load environment variables
load_dotenv()

# Function to randomly select an API key
def get_random_api_key():
    gemini_api_keys = os.getenv("GEMINI_API_KEYS")
    if gemini_api_keys:
        api_keys = gemini_api_keys.split(",")
        selected_key = random.choice(api_keys)  # Randomly select one key
        logging.info(f"Selected API key: {selected_key}")  # Log the selected key
        return selected_key
    else:
        raise ValueError("GEMINI_API_KEYS not set in the .env file.")

# Configure Gemini API with a random API key
api_key = get_random_api_key()
genai.configure(api_key=api_key)


# Create FastAPI app instance
app = FastAPI()

#CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # can restrict this to specific methods like ["GET", "POST"]
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "Welcome to my API!"}

# Define the request model
class GeneratePromptsRequest(BaseModel):
    content_type: str
    audience_type: str
    delivery_method: str
    content_theme: str
    target_industry: str

class PromptRequest(BaseModel):
    prompt: str


# Function to extract numbered points from response
def extract_numbered_points(text):
    try:
        json_start = text.find('```json\n[')
        json_end = text.find(']\n```')
        if json_start != -1 and json_end != -1:
            json_str = text[json_start + 7:json_end + 1]
            return json.loads(json_str)
        else:
            return [text.strip()]
    except json.JSONDecodeError as e:
        return [f"Malformed JSON: {e}"]
    except Exception as e:
        return [f"Unexpected error: {e}"]

# Endpoint to generate content creation prompts
@app.post("/generate_prompts")
async def generate_prompts(request: GeneratePromptsRequest):
    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config={
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            },
            system_instruction="You are expert at prompt engineering and your goal is to write prompts helping the trainers to create professional and relevant content."
        )

        # Extracting parameters from request
        content_type = request.content_type
        audience_type = request.audience_type
        delivery_method = request.delivery_method
        content_theme = request.content_theme
        target_industry = request.target_industry

        prompt_template = f"""Generate 4 content creation prompts to help trainers generate content based on the following inputs:
            \nContent Type - {content_type}
            \nAudience Type - {audience_type}
            \nDelivery Method - {delivery_method}
            \nContent Theme - {content_theme}
            \nTarget Industry - {target_industry}
            \nFor each prompt, provide:
            1. A detailed version for content generation
            2. A brief 2-3 line summary for display
            \nPlease format your response as a JSON array where each element is an object with 'detailed_prompt' and 'summary' keys."""

        chat_session = model.start_chat(
            history=[{
                "role": "user",
                "parts": [prompt_template]
            }]
        )

        response = chat_session.send_message("Generate the content creation prompts.")
        prompts = extract_numbered_points(response.text)

        if not prompts:
            return [{"detailed_prompt": "Error extracting prompts.", "summary": "Error"}]

        return prompts

    except Exception as e:
        logging.error(f"Error generating prompts: {e}")
        return [{"detailed_prompt": f"Error: {e}", "summary": "Error"}]

# Function to ask a specific prompt to Gemini API
@app.post("/ask-gemini/")
def ask_gemini(prompt_request: PromptRequest):
    try:
        selected_prompt = prompt_request.prompt

        # Send the selected prompt to Gemini
        chat_session = genai.GenerativeModel(
            model_name="gemini-1.5-pro"
        ).start_chat(
            history=[{
                "role": "user",
                "parts": [selected_prompt]
            }]
        )

        response = chat_session.send_message(selected_prompt)
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")
