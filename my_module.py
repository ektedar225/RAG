import os
import hashlib
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pinecone import Pinecone, ServerlessSpec
import uuid
import zipfile
from pypdf.errors import PdfReadError

def create_pinecone_index_if_not_exists(index_name="hackrx-index-2"):
    """
    Checks if a Pinecone index exists, and creates it if it doesn't.
    """
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,  # Dimension for OpenAI's text-embedding-ada-002
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

def search_pinecone_memory(namespace: str, query: str, embeddings, threshold: float = 0.95, index_name="hackrx-index-2"):
    """
    Searches for a similar question in the Pinecone Q&A cache.
    """
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(index_name)

    query_vector = embeddings.embed_query(query)

    try:
        results = index.query(
            namespace=namespace,
            vector=query_vector,
            top_k=1,
            include_metadata=True
        )
        if results.matches and results.matches[0].score >= threshold:
            return results.matches[0].metadata.get("answer")
    except Exception:
        # Namespace might not exist yet
        return None
    return None

def save_to_pinecone_memory(namespace: str, question: str, answer: str, embeddings, index_name="hackrx-index-2"):
    """
    Saves a question-answer pair to the Pinecone Q&A cache.
    """
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(index_name)

    question_vector = embeddings.embed_query(question)
    # Use a unique ID for each vector
    vector_id = str(uuid.uuid4())

    index.upsert(
        vectors=[
            {
                "id": vector_id,
                "values": question_vector,
                "metadata": {"question": question, "answer": answer}
            }
        ],
        namespace=namespace
    )


# This function checks if a namespace already exists and has vectors.
def check_if_namespace_exists(namespace: str, index_name="hackrx-index-2"):
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(index_name)
    try:
        index_stats = index.describe_index_stats()
        return namespace in index_stats.namespaces and index_stats.namespaces[namespace].vector_count > 0
    except Exception:
        return False


def load_documents(folder_path: str):
    docs = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        try:
            # Try to load as a PDF first
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
        except (PdfReadError, UnicodeDecodeError):
            # If it fails, check if it's a zip file
            if zipfile.is_zipfile(file_path):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    # Create a directory to extract files
                    extract_path = os.path.join(folder_path, "extracted")
                    if not os.path.exists(extract_path):
                        os.makedirs(extract_path)
                    zip_ref.extractall(extract_path)

                    # Now, look for PDFs in the extracted files
                    for extracted_file in os.listdir(extract_path):
                        if extracted_file.endswith(".pdf"):
                            extracted_file_path = os.path.join(extract_path, extracted_file)
                            loader = PyPDFLoader(extracted_file_path)
                            docs.extend(loader.load())
    return docs

def split_documents(documents, embeddings):
    text_splitter = SemanticChunker(embeddings)
    return text_splitter.split_documents(documents)

def build_vectorstore(chunks, embeddings, namespace: str, index_name="hackrx-index-2"):
    # This will now only be called if the document is not cached.
    vectorstore = PineconeVectorStore.from_documents(
        chunks,
        embedding=embeddings,
        index_name=index_name,
        namespace=namespace
    )
    return vectorstore

def get_existing_vectorstore(namespace: str, embeddings, index_name="hackrx-index-2"):
    # This function is used when the document is already in Pinecone.
    # It initializes the vector store without adding any new documents.
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace
    )
    return vectorstore

def build_qa_chain(vectorstore, namespace: str):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10, "namespace": namespace}
    )

    answer_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""

"You are to act as a specialized Information Extraction Bot.Your function is to analyze the document I provide and answer my specific questions. You must operate under the following directives:

1. Base All Answers on the Provided Document: Your knowledge base is the context provided.

2. No Evasive Answers: You are forbidden from using phrases like "refer to the policy document," "for more details, see section X," or "the document states...". Your job is to be the document expert, so you must provide the answer directly.

3. Be Direct and Factual: Answer the question directly and concisely, using a formal and factual tone. Do not add conversational introductions or conclusions.

4. Prioritize and Include All Numerical Data: This is a primary directive. When you formulate an answer, you must actively search the document for any and all numerical information related to that answer. You must integrate this data directly into your response. This includes, but is not limited to: •⁠ ⁠Time Periods: waiting periods (in days, months, or years), grace periods. •⁠ ⁠Monetary Values: coverage limits, sub-limits, caps on expenses, deductibles. •⁠ ⁠Percentages: co-payments, discounts (like No Claim Discount). •⁠ ⁠Quantities: number of treatments, deliveries, or check-ups covered.



Context:

{context}



Question: {question}

"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": answer_prompt}
    )
