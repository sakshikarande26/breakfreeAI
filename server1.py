from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from phi.agent import Agent
from phi.model.groq import Groq
from fastapi.middleware.cors import CORSMiddleware
import json

load_dotenv()
api_key = os.getenv('GROQ_API_KEY')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    content_type: str
    audience_type: str
    delivery_method: str
    content_theme: str
    target_industry: str


class ContentRequest(BaseModel):
    prompts: str  

@app.get("/")
async def read_root():
    return {"message": "Welcome to my API!"}


@app.post("/generate_prompts/")
async def generate_prompts(request: PromptRequest):
    try:
        prompt_agent = Agent(
            name="Prompt Generation Agent",
            model=Groq(id="llama-3.3-70b-versatile"),
            markdown=True,
            instructions=f"""
                Generate 4 distinct content creation prompts for trainers to help them generate content based on the following user inputs:
                - Content Type: {request.content_type}
                - Audience Type: {request.audience_type}
                - Delivery Method: {request.delivery_method}
                - Content Theme: {request.content_theme}
                - Target Industry: {request.target_industry}
                
                For each prompt, provide:
                1. A **detailed prompt** (without a title) that explains the type of content to create.
                2. A **short 2-3 sentence summary** describing the essence of the prompt including the key details from the user inputs above.
                
                The final output should be in the format:
                Prompt 1: (Detailed prompt here)
                Summary 1: (Summary for the first prompt here)
                Prompt 2: (Detailed prompt here)
                Summary 2: (Summary for the second prompt here)
                Prompt 3: (Detailed prompt here)
                Summary 3: (Summary for the third prompt here)
                Prompt 4: (Detailed prompt here)
                Summary 4: (Summary for the fourth prompt here)
                
                Ensure the response is structured properly so it can be parsed into JSON format.
            """,
            description="You are an AI agent that helps trainers generate tailored content for employee training sessions. Use the inputs provided to create structured prompts and summaries."
        )

        # Run the agent to generate the response
        response = prompt_agent.run("Generate 4 distinct prompts with summaries.")

        # Parse the response into structured key-value pairs
        generated_prompts = response.content.strip().split("\n")

        return generated_prompts

    except Exception as e:
        return {"error": str(e)}


@app.post("/generate_content/")
async def generate_content(request: ContentRequest):
    try:
        content_agent = Agent(
            name="Content Generation Agent",
            model=Groq(id="llama-3.3-70b-versatile"),
            markdown=True,
            instructions=f"Generate detailed content based on the following prompt: {request.prompts}",
            description="You are an agent that generates comprehensive and structured training content based on the provided detailed prompt."
        )

        response = content_agent.run(f"Generate detailed content based on the following prompt: {request.prompts}")
        return {"content": response.content.split("\n")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
