from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import google.generativeai as genai
import json

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create FastAPI app instance
app = FastAPI()

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

        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        f"""Generate 4 content creation prompts to help trainers generate content based on the following inputs:
                        \nContent Type - {request.content_type}\nAudience Type - {request.audience_type}\nDelivery Method - {request.delivery_method}
                        \nContent Theme - {request.content_theme}\nTarget Industry - {request.target_industry}
                        \nPlease format your response as a JSON array."""
                    ],
                }
            ]
        )
        response = chat_session.send_message("Generate the content creation prompts.")
        prompts = extract_numbered_points(response.text)

        if not prompts:
            return {"error": "Error extracting prompts. Please check the API response format."}

        return {"prompts": prompts}

    except Exception as e:
        return {"error": f"Error generating prompts: {e}"}

# Function to ask a specific prompt to Gemini API
@app.post("/ask-gemini/")
async def ask_gemini(prompt_request: PromptRequest):
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

# Add a root route to avoid 404 errors for "/"
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Gemini API integration!"}
