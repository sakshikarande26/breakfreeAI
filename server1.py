from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from phi.agent import Agent
from phi.model.groq import Groq
from fastapi.middleware.cors import CORSMiddleware

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


class ContentRequest(PromptRequest):
    prompt: str  

@app.get("/")
async def read_root():
    return {"message": "Welcome to my API!"}

@app.post("/generate_prompts/")
async def generate_prompts(request: PromptRequest):
    """Generate 4 distinct training content prompts based on user input."""
    try:
        prompt_agent = Agent(
            name="Prompt Generation Agent",
            model=Groq(id="llama-3.3-70b-versatile"),
            markdown=True,
            instructions=f"Generate 4 distinct content creation prompts to help trainers who train employees generate content based on the following user inputs: Content Type - {request.content_type}, Audience Type - {request.audience_type}, Delivery Method - {request.delivery_method}, Content Theme - {request.content_theme}, Target Industry - {request.target_industry}. For each prompt, provide a detailed version for content generation without a title.",
            description="You are an AI agent specializing in helping trainers develop tailored content for employee training sessions."
        )

        response = prompt_agent.run("generate 4 distinct prompts that can be used to generate relevant detailed content without title")
        
        return {"prompts": response.content.split("\n")}  

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize_prompts/")
async def summarize_prompts(request: SummarizeRequest):
    """Generate summaries for given prompts while maintaining the context."""
    try:
        summary_agent = Agent(
            name="Prompt Summary Generation Agent",
            model=Groq(id="llama-3.3-70b-versatile"),
            markdown=True,
            instructions=f"Generate 4 individual summaries of 2-3 sentences for each of the following prompts while maintaining the context. The user selections are: Content Type - {request.content_type}, Audience Type - {request.audience_type}, Delivery Method - {request.delivery_method}, Content Theme - {request.content_theme}, Target Industry - {request.target_industry}. Prompts: {', '.join(request.prompts)}",
            description="You are an agent that summarizes prompts while preserving the context and incorporating user selections."
        )

        response = summary_agent.run("Generate a concise summary for each of the provided prompts and give heading as summary (1/2/3/4): (generated summaries)")
        
        return {"summaries": response.content.split("\n")}  

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/generate_content/")
async def generate_content(request: ContentRequest):
    try:
        content_agent = Agent(
            name="Content Generation Agent",
            model=Groq(id="llama-3.3-70b-versatile"),
            markdown=True,
            instructions=f"Generate detailed content based on the following prompt: {request.prompt}",
            description="You are an agent that generates comprehensive and structured training content based on the provided detailed prompt."
        )

        response = content_agent.run(f"Generate detailed content based on the following prompt: {request.prompt}")
        return {"content": response.content.split("\n")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
