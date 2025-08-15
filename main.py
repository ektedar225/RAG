from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import List
import os
import httpx
from dotenv import load_dotenv
import hashlib
import asyncio

# Import the new functions from my_module
from my_module import (
    load_documents,
    split_documents,
    build_vectorstore,
    get_existing_vectorstore,
    check_if_namespace_exists,
    build_qa_chain,
    search_pinecone_memory, # <-- Updated
    save_to_pinecone_memory,   # <-- Updated
    create_pinecone_index_if_not_exists
)
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Create the Pinecone index if it doesn't exist
create_pinecone_index_if_not_exists()


app = FastAPI()

embeddings = OpenAIEmbeddings()

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

@app.post("/hackrx/run")
async def run_hackrx(request: HackRxRequest, authorization: str = Header(...)):

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header")
    
    print(f"Received request payload: {request.dict()}")

    # Namespace for document chunks
    doc_namespace = hashlib.sha256(request.documents.encode()).hexdigest()
    # Separate namespace for Q&A cache
    qa_namespace = f"{doc_namespace}-qa"


    # Check if the document is already processed and stored
    if await run_in_threadpool(check_if_namespace_exists, doc_namespace):
        vectorstore = await run_in_threadpool(get_existing_vectorstore, doc_namespace, embeddings)
    else:
        # If not cached, download and process the PDF
        os.makedirs("SampleDocs", exist_ok=True)
        pdf_path = os.path.join("SampleDocs", f"{doc_namespace}.pdf")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(request.documents)
                response.raise_for_status()
            with open(pdf_path, "wb") as f:
                f.write(response.content)
        except httpx.RequestError as e:
            raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred during PDF download: {str(e)}")

        try:
            docs = await run_in_threadpool(load_documents, "SampleDocs/")
            chunks = await run_in_threadpool(split_documents, docs, embeddings)
            vectorstore = await run_in_threadpool(build_vectorstore, chunks, embeddings, namespace=doc_namespace)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


    try:
        qa_chain = await run_in_threadpool(build_qa_chain, vectorstore, namespace=doc_namespace)
        answers = []

        for q in request.questions:
            try:
                async def answer_question():
                    # 1. Try to retrieve answer from Pinecone memory
                    cached_answer = await run_in_threadpool(search_pinecone_memory, qa_namespace, q, embeddings)
                    if cached_answer:
                        return cached_answer

                    # 2. Else, run the full RAG pipeline
                    response = qa_chain({"query": q})
                    answer = response["result"]

                    # 3. Save new Q-A to Pinecone memory
                    await run_in_threadpool(save_to_pinecone_memory, qa_namespace, q, answer, embeddings)
                    
                    return answer

                answer = await asyncio.wait_for(answer_question(), timeout=55.0)
                answers.append(answer)

            except asyncio.TimeoutError:
                raise HTTPException(status_code=408, detail=f"Processing question timed out: {q}")


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")
    
    #await asyncio.sleep(10)

    return {
        "answers": answers
    }
