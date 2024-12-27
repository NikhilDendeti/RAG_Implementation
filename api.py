from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from main import execute_rag_pipeline

# Create FastAPI instance
app = FastAPI()

# Allow CORS from specific origins (replace with your frontend URL)
origins = [
    "http://0.0.0.0:8000",  # For your local frontend
    "http://localhost:8000",  # In case of accessing from localhost
    "http://192.168.1.29:8000",  # Your specific local network IP
]

# Add CORSMiddleware to the FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request body schema
class QuestionRequest(BaseModel):
    question: str
    num_responses: int

# API Endpoint to process questions
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        # Assume the PDF is preloaded in your backend
        pdf_path = "Insurance Industry _ Report.pdf"

        # Execute the RAG pipeline
        result = execute_rag_pipeline(pdf_path, request.question, k=request.num_responses)

        # Respond with the generated answer
        return JSONResponse(content={
            "Question": request.question,
            "Answer": result["Answer"],
            "RelevantDocuments": result["Relevant Documents"],
            "HypotheticalDocument": result["Hypothetical Document"],
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
