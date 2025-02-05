from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from phi.agent import Agent, RunResponse
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
    prompt_agent = Agent(
        name="Prompt Generation Agent",
        model=Groq(id="llama-3.3-70b-versatile"),
        markdown=True,
        instructions=f"""
            Generate strictly 4 distinct content creation prompts for trainers to help them generate content based on the following user inputs:
            - Content Type: {request.content_type}
            - Audience Type: {request.audience_type}
            - Delivery Method: {request.delivery_method}
            - Content Theme: {request.content_theme}
            - Target Industry: {request.target_industry}
            
            For each prompt, provide:
            1. A **detailed prompt** (without a title) that explains the type of content to create based on the user input.
            2. A **short 2-3 sentence version of the above prompt summarising it and describing the essence of the prompt including the key details from the user inputs above. Generate the content in such a way that viewing either of them individually can convey the same idea (meaning dont start the summary with "this prompt states that.." etc). Such that it will look like 
               Prompt:
               Summary:
            
            The final output should be in the key: value pairs format:
            "Prompt 1: (Detailed prompt here)
            Summary 1: (Summary for the first prompt here)"
            "Prompt 2: (Detailed prompt here)
            Summary 2: (Summary for the second prompt here)"
            "Prompt 3: (Detailed prompt here)
            Summary 3: (Summary for the third prompt here)"
            "Prompt 4: (Detailed prompt here)
            Summary 4: (Summary for the fourth prompt here)"

            
        """,
        description="You are an AI agent that helps trainers generate tailored content for employee training sessions. Use the inputs provided to create prompts and summaries suitable for the given context without a heading like '### Prompts and Summaries for Training Modules on Leadership and Management'"
    )

    response = prompt_agent.run("generate 4 distinct prompts with their individual summaries that can be used to generate relevant detailed content without title ")
    structured_response = response.content.split("\n")
    
    result = {}
    
    # Adjusted loop to correctly index keys and strip prefixes
    for i in range(0, len(structured_response), 3):
        if structured_response[i] != "" and structured_response[i + 1] != "":
            key = f"key_{(i // 3) + 1}"
            
            # Clean up prompts and summaries by removing any prefix
            detailed_prompt = structured_response[i].replace("* ", "").strip()
            summary_text = structured_response[i + 1].replace("* ", "").strip()
            
            # Remove prefixes from detailed_prompt and summary_text
            if ": " in detailed_prompt:
                detailed_prompt = detailed_prompt.split(": ", 1)[1]
                
            if ": " in summary_text:
                summary_text = summary_text.split(": ", 1)[1]
            
            result[key] = {
                "prompt": detailed_prompt,
                "summary": summary_text
            }
    
    return result


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
    
