import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import AIMessage, SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser

# Load API Key from .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize FastAPI
app = FastAPI()

# Initialize ChatGroq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key,
    temperature=0.7,
    top_p=1,
    max_retries=2,
)

# Conversation memory store (can be replaced with Redis for persistence)
conversation_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

class PromptRequest(BaseModel):
    content_type: str
    audience_type: str
    delivery_method: str
    content_theme: str
    target_industry: str

class ChatRequest(BaseModel):
    user_input: str

# Function to create a prompt template
def create_prompt_template():
    return """ 
    You are an AI agent that helps trainers create tailored content for employee training sessions.
    Generate 4 distinct content creation prompts for trainers based on the following user inputs:
    - Content Type: {content_type}
    - Audience Type: {audience_type}
    - Delivery Method: {delivery_method}
    - Content Theme: {content_theme}
    - Target Industry: {target_industry}

    For each prompt, provide:
    1. A detailed prompt (without a title) that explains the type of content to create.
    2. A short 2-3 sentence version summarizing the above prompt.

    ### Return the output as a JSON object:
    ```json
    {{
      "prompts": [
        {{
          "prompt 1": "Detailed prompt 1 here",
          "summary 1": "Summary for prompt 1 here"
        }},
        {{
          "prompt 2": "Detailed prompt 2 here",
          "summary 2": "Summary for prompt 2 here"
        }},
        {{
          "prompt 3": "Detailed prompt 3 here",
          "summary 3": "Summary for prompt 3 here"
        }},
        {{
          "prompt 4": "Detailed prompt 4 here",
          "summary 4": "Summary for prompt 4 here"
        }}
      ]
    }}
    """

# Function to generate initial prompts
def generate_initial_prompts(inputs):
    template = create_prompt_template()
    prompt_template = PromptTemplate(
        input_variables=["content_type", "audience_type", "delivery_method", "content_theme", "target_industry"],
        template=template,
    )

    output_parser = JsonOutputParser()
    first_chain = LLMChain(llm=llm, prompt=prompt_template, output_parser=output_parser)

    try:
        response = first_chain.run(inputs)
        return json.loads(json.dumps(response))  # Ensure valid JSON
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoint to generate initial prompts
@app.post("/generate-prompts")
async def generate_prompts(request: PromptRequest):
    inputs = request.model_dump()
    response = generate_initial_prompts(inputs)
    return response


# Function to initialize conversation memory with initial prompts
def initialize_memory(initial_response):
    conversation_memory.clear()  # Clear old memory
    conversation_memory.save_context(
        {"input": "Generated Initial Prompts"},
        {"output": json.dumps(initial_response)}
    )

# Create chat prompt template
def get_chat_prompt():
    return ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful AI assistant for employee training. Help users create effective content and answer their questions patiently."),
        MessagesPlaceholder(variable_name="chat_history"),
        AIMessage(content="Ask me anything related to training, and I'll generate responses based on our conversation history.")
    ])

# API Endpoint for Chat Assistant
@app.post("/chat")
async def chat_with_assistant(request: ChatRequest):
    user_input = request.user_input.strip()
    
    # If no input, return an error
    if not user_input:
        raise HTTPException(status_code=400, detail="User input cannot be empty.")

    # If first message, initialize memory
    if "initial_response" in request and request.initial_response:
        initialize_memory(request.initial_response)

    # Handle special commands
    if user_input.lower() == "clear memory":
        conversation_memory.clear()
        return {"response": "Conversation memory has been cleared."}

    # Create LLMChain with conversation memory
    chain = LLMChain(
        llm=llm,
        prompt=get_chat_prompt(),
        memory=conversation_memory,
    )

    try:
        response = chain.predict(input=user_input)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return {"message": "Welcome to my API!"}
