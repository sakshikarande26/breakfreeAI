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

class ContentRequest(BaseModel):
    prompt: str  

@app.get("/")
async def read_root():
    return {"message": "Welcome to my API!"}

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
