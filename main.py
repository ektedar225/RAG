# from fastapi import FastAPI, Request, Header, HTTPException
# from pydantic import BaseModel
# from typing import List
# import os
# import requests
# from dotenv import load_dotenv

# from my_module import load_documents, split_documents, build_qa_chain
# from langchain_community.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings

# # Load environment variables (e.g., OpenAI API key)
# load_dotenv()

# app = FastAPI()

# # Define the request body model
# class HackRxRequest(BaseModel):
#     documents: str  # URL to a PDF
#     questions: List[str]

# @app.post("/hackrx/run")
# async def run_hackrx(request: HackRxRequest, authorization: str = Header(...)):
#     # Validate Authorization header
#     if not authorization.startswith("Bearer "):
#         raise HTTPException(status_code=401, detail="Invalid Authorization header")

#     # Step 1: Download PDF
#     os.makedirs("SampleDocs", exist_ok=True)
#     pdf_path = os.path.join("SampleDocs", "input.pdf")
#     try:
#         response = requests.get(request.documents)
#         with open(pdf_path, "wb") as f:
#             f.write(response.content)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")

#     # Step 2: Process documents and build vectorstore
#     try:
#         docs = load_documents("SampleDocs/")
#         chunks = split_documents(docs)
#         embeddings = OpenAIEmbeddings()
#         vectorstore = FAISS.from_documents(chunks, embeddings)
#         qa_chain = build_qa_chain(vectorstore)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

#     # Step 3: Answer the questions
#     try:
#         answers = []
#         for q in request.questions:
#             result = qa_chain({"query": q})
#             answers.append(result["result"])
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")

#     return {"answers": answers}


# main.py

from fastapi import FastAPI, Request, Header, HTTPException, status
from pydantic import BaseModel
from typing import List
import os
import requests
import shutil # For safely removing directories
from dotenv import load_dotenv

# Import functions from your local module
from my_module import load_documents, split_documents, build_qa_chain, get_gemini_embeddings
from langchain_community.vectorstores import FAISS

# Load environment variables (e.g., GOOGLE_API_KEY)
load_dotenv()

app = FastAPI(
    title="HackRx Document Q&A API",
    description="API for processing PDF documents and answering questions using Gemini LLM."
)

# Define the request body model for the API endpoint
class HackRxRequest(BaseModel):
    documents: str  # URL to a PDF file
    questions: List[str] # List of questions to ask about the document

@app.post("/hackrx/run", status_code=status.HTTP_200_OK)
async def run_hackrx(request: HackRxRequest, authorization: str = Header(...)):
    """
    Processes a PDF document from a URL and answers a list of questions using Gemini LLM.

    Args:
        request (HackRxRequest): Contains the URL to the PDF and a list of questions.
        authorization (str): Bearer token for API authorization.

    Returns:
        dict: A dictionary containing the answers to the questions.
    """
    # Validate Authorization header for basic API security
    # In a real application, this would involve more robust token validation (e.g., OAuth2)
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization header. Expected 'Bearer <token>'.")
    
    # You might want to compare the token with an expected value from environment variables
    # For example:
    # EXPECTED_API_TOKEN = os.getenv("API_TOKEN")
    # if authorization.split(" ")[1] != EXPECTED_API_TOKEN:
    #     raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized token.")

    # Define a temporary directory to store the downloaded PDF
    temp_dir = "SampleDocs_temp"
    pdf_path = os.path.join(temp_dir, "input.pdf")

    try:
        # Step 1: Download PDF from the provided URL
        os.makedirs(temp_dir, exist_ok=True) # Create the temporary directory if it doesn't exist
        print(f"Attempting to download PDF from: {request.documents}")
        response = requests.get(request.documents, stream=True) # Use stream=True for large files
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        with open(pdf_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"PDF downloaded successfully to: {pdf_path}")

        # Step 2: Process documents and build vectorstore
        # This involves loading the PDF, splitting it into chunks,
        # generating embeddings for the chunks, and storing them in FAISS.
        print("Loading and processing documents...")
        docs = load_documents(temp_dir)
        if not docs:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No documents found or loaded from the PDF.")
        
        chunks = split_documents(docs)
        if not chunks:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No text chunks could be extracted from the PDF.")
        
        # Get Gemini embeddings
        embeddings = get_gemini_embeddings()
        print("Generating embeddings and building FAISS vectorstore...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        print("FAISS vectorstore built.")

        # Build the QA chain using the vectorstore and Gemini LLM
        qa_chain = build_qa_chain(vectorstore)
        print("QA chain built.")

        # Step 3: Answer the questions
        print("Answering questions...")
        answers = []
        for i, q in enumerate(request.questions):
            print(f"  Answering question {i+1}: '{q}'")
            try:
                result = qa_chain({"query": q})
                answers.append(result.get("result", "I don't know.")) # Use .get to safely access 'result'
                print(f"  Answer {i+1}: {result.get('result', 'N/A')}")
            except Exception as qa_e:
                print(f"Error answering question '{q}': {qa_e}")
                answers.append(f"Error answering this question: {qa_e}")
        print("All questions answered.")

        return {"answers": answers}

    except requests.exceptions.RequestException as req_e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to download PDF from URL: {req_e}")
    except Exception as e:
        # Catch any other unexpected errors during the process
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Processing failed: {str(e)}")
    finally:
        # Clean up the temporary directory and downloaded PDF file
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as cleanup_e:
                print(f"Error cleaning up temporary directory {temp_dir}: {cleanup_e}")

