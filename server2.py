import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import re

# Import LangChain-related components
from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

PROMPT_TEMPLATE_1 = """1. Objective:
        You are responsible for analyzing the content of a given CSV file and 
        determining whether it qualifies as a lesson plan. Your response should be a "Yes" (If it a lesson plan)
        or "No" if it is not a Lesson plan. 
        
        2. Process Overview:

        2.1. Read the CSV File
            a. Parse the content and extract all relevant text fields.

        2.2. Identify Lesson Plan Components
            a. Compare the extracted content against the required topics:
                i. Training Title & Audience
                ii. Learning Objectives
                iii. Course Duration & Format
                iv. Instructional Strategy
                v. Course Outline (Modules & Topics)
                vi. Training Activities & Exercises
                vii. Assessment Methods
                viii. Resources Required
                ix. Evaluation Criteria (Success Metrics)
                x. Delivery Plan & Facilitator Guidelines
                xi. Follow-Up & Retention Strategies

        2.3. Determine Lesson Plan Eligibility
            a. Calculate the percentage of required topics present.
            b. If at least 80% (i.e., 9 out of 11) of the topics are included, classify the content as a lesson plan.

        2.4. Respond "Yes" or "No" with your reasoning   
            
        3. Implementation Notes:
            a. Ensure the CSV file is properly parsed, handling different delimiters and encodings.
            b. Use keyword matching or natural language processing (NLP) techniques to detect topic presence.
            c. If a topic is ambiguous, apply a best-effort approach to determine relevance.
            d. Maintain a log of decisions for auditing and debugging purposes."""

PROMPT_TEMPLATE_2 = """Your goal is to evaluate the document using the following pointers -
        Step 1: Understanding the Evaluation Process
        1. Your goal is to use the additional context which contains the criteria for evaluation, along with specific instructions on how to assess each item."        
        2. Each criterion may have an associated score
        3. Responses should be determined based on the details provided in the input file."

        Step 2: Evaluating Each Criterion 
        1. Read the provided input file carefully.
        2. Identify the relevant information needed for evaluation.
        3. Determine whether the input meets each specified requirement.
        4. Assign the corresponding score 

        Step 3: Calculating the Total Score
        1. Sum up all the awarded scores.
        2. Ensure the final total reflects the correct scoring based on responses.

        Step 4: Final Review and Submission
        1. Double-check that all criteria have been marked.
        2. Verify calculations to ensure accuracy.
        3. Submit the final evaluation report with score and reasoning as the output.
        
        ### Additional Context

        Instructions Based on Input File:

        1. Training Title & Audience (10 points)
        a. Training title is clear and relevant (5 points).
        b. Target audience is clearly defined, including prerequisites (5 points).

        2. Learning Objectives (15 points)
        a. Objectives are clearly stated and measurable (8 points).
        b. Objectives align with organizational goals (7 points).

        3. Course Duration & Format (10 points)
        a. Total training duration is specified (5 points).
        b. Delivery mode is clearly mentioned (5 points).

        4. Instructional Strategy (15 points)
        a. Training has a well-structured content flow (5 points).
        b. Teaching methods are engaging and interactive (5 points).
        c. Training follows adult learning principles (5 points).

        5. Assessment & Feedback Mechanism (10 points)
        a. Clear assessment criteria are defined (5 points).
        b. Feedback mechanisms are in place for trainees (5 points).

        6. Resources & Materials (10 points)
        a. All necessary training materials are available (5 points).
        b. Materials are clearly outlined and well-structured (5 points).

        7. Trainer Qualifications & Delivery (10 points)
        a. Trainer's credentials and experience meet required standards (5 points).
        b. Trainer engages learners effectively (5 points).

        8. Interactivity & Engagement (10 points)
        a.  Training includes interactive elements (5 points).
        b. Learners are actively engaged throughout the session (5 points).

        9. Post-Training Support & Application (10 points)
        a. Follow-up resources or support are provided (5 points).
        b. Training includes guidance on applying learned skills in the workplace (5 points).

        10. Delivery Plan & Facilitator Guidelines (10 points)
        a. Structured delivery plan is provided (5 points).
        b. Facilitator guidelines offer clear instructions for conducting the training (5 points).

        11. Follow-Up & Retention Strategies (10 points)
        a. Mechanisms are in place to reinforce learning post-training (5 points).
        b. Retention strategies (e.g., refresher sessions, job aids, mentorship programs) are included (5 points).
        ###"""

PROMPT_TEMPLATE_3 = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

# ----------------- Global Setup ----------------- #
file_processed=False
PDF_STORAGE_PATH = '/Users/sakshikarande/Desktop/langchain_evaluate/'
EMBEDDING_MODEL = GPT4AllEmbeddings()
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = init_chat_model(
    model="deepseek-r1-distill-llama-70b",
    model_provider="groq",
    temperature=0,
    top_p=1
)

# ----------------- Utility Functions ----------------- #
def save_uploaded_file(uploaded_file: UploadFile) -> str:
    """Saves the uploaded file locally and returns its file path."""
    file_location = os.path.join(PDF_STORAGE_PATH, uploaded_file.filename)
    with open(file_location, "wb") as f:
        f.write(uploaded_file.file.read())
    return file_location

def load_pdf_documents(file_path: str):
    """Loads the document using the UnstructuredCSVLoader."""
    document_loader = UnstructuredCSVLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    """Splits the raw document into manageable text chunks."""
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    """Adds the document chunks to the in-memory vector store."""
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query: str):
    """Performs a similarity search on the vector store for the given query."""
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def process_file(uploaded_file: UploadFile):
    """Handles file saving, loading, chunking, and indexing."""
    file_path = save_uploaded_file(uploaded_file)
    raw_docs = load_pdf_documents(file_path)
    document_chunks = chunk_documents(raw_docs)
    index_documents(document_chunks)

def default_answer():
    """Generates the yes/no lesson plan evaluation with explanation."""
    conversation_prompt = PromptTemplate.from_template(PROMPT_TEMPLATE_1)
    response_chain_1 = conversation_prompt | LANGUAGE_MODEL
    user_query_1 = "Please evaluate if the doc is a Lesson plan or not."
    relevant_docs = find_related_documents(user_query_1)
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    response_1 = response_chain_1.invoke({
        "user_query": user_query_1,
        "document_context": context_text
    })
    return response_1.content

def default_answer_2():
    """Generates the overall scoring evaluation with a detailed breakdown."""
    conversation_prompt_2 = PromptTemplate.from_template(PROMPT_TEMPLATE_2)
    response_chain_2 = conversation_prompt_2 | LANGUAGE_MODEL
    user_query_2 = "Please evaluate the document."
    relevant_docs = find_related_documents(user_query_2)
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    response_2 = response_chain_2.invoke({
        "user_query": user_query_2,
        "document_context": context_text
    })
    return response_2.content


# ----------------- Endpoints ----------------- #

# Endpoint 1: File Upload
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Accepts a PDF or CSV file upload, processes the file (saves, loads, chunks, and indexes),
    and returns a confirmation.
    """
    if file.content_type not in ["application/pdf", "text/csv"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a PDF or CSV file."
        )
    
    process_file(file)
    
    global file_processed
    file_processed = True
    
    return JSONResponse(content={"message": "File processed successfully."})

@app.post("/evaluate/done")
async def evaluate_done():
    """
    Returns a JSON response containing:
      - lesson_plan_evaluation: Yes/No decision with explanation.
      - overall_score: The total score extracted from the evaluation.
    """
    if not file_processed:
        raise HTTPException(
            status_code=400,
            detail="No file has been processed. Please upload a file first."
        )
    
    lesson_plan_eval = default_answer()
    overall_score_eval = default_answer_2()

    # Extract "Total Score: 110/110"
    match = re.search(r"Total Score: (\d+/\d+)", overall_score_eval)

    # Extract only the score part (e.g., "110/110")
    total_score = match.group(1) if match else "N/A"


    result = {
        "lesson_plan_evaluation": lesson_plan_eval,
        "overall_score": total_score
    }
    return JSONResponse(content=result)


@app.get("/")
async def read_root():
    return {"message": "Welcome to my API!"}



