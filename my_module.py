# import os
# from langchain_community.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import RetrievalQA
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain.prompts import PromptTemplate

# def load_documents(folder_path):
#     docs = []
#     for file in os.listdir(folder_path):
#         if file.endswith(".pdf"):
#             loader = PyPDFLoader(os.path.join(folder_path, file))
#             docs.extend(loader.load())
#     return docs

# def split_documents(documents):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         separators=["\n\n", "\n", ".", " ", ""]
#     )

#     chunks = text_splitter.split_documents(documents)

#     for chunk in chunks:
#         text = chunk.page_content.lower()
#         if "coverage" in text or "covered" in text:
#             section = "coverage"
#         elif any(term in text for term in ["exclusion", "excluded", "not covered"]):
#             section = "exclusions"
#         elif any(term in text for term in ["term", "definition", "meaning"]):
#             section = "general"
#         else:
#             section = "other"

#         chunk.metadata["section"] = section

#     return chunks

# def build_qa_chain(vectorstore):
#     llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

#     base_retriever = vectorstore.as_retriever(
#         search_type="similarity",
#         search_kwargs={"k": 5}
#     )

#     retriever = MultiQueryRetriever.from_llm(
#         retriever=base_retriever,
#         llm=llm
#     )

#     prompt_template = PromptTemplate(
#         input_variables=["context", "question"],
#         template="""
# You are a helpful assistant. Use only the following context to answer the question. If the answer is not in the context, say "I don't know."

# Context:
# {context}

# Question: {question}
# """
#     )

#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         return_source_documents=True,
#         chain_type="stuff",
#         chain_type_kwargs={"prompt": prompt_template}
#     )

#     return qa_chain


# my_module.py

import os
from langchain_community.vectorstores import FAISS
# Import Google Generative AI components for Chat and Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate

# It's good practice to load dotenv here too, in case my_module is run independently
# or for clarity, though main.py also loads it.
from dotenv import load_dotenv
load_dotenv()

def load_documents(folder_path):
    """
    Loads all PDF documents from a specified folder.
    """
    docs = []
    if not os.path.exists(folder_path):
        print(f"Warning: Folder '{folder_path}' does not exist.")
        return docs
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            try:
                loader = PyPDFLoader(file_path)
                docs.extend(loader.load())
                print(f"Loaded {len(loader.pages)} pages from {file}.")
            except Exception as e:
                print(f"Error loading PDF {file_path}: {e}")
    return docs

def split_documents(documents):
    """
    Splits documents into smaller chunks and adds a 'section' metadata based on content.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    for chunk in chunks:
        # Convert page content to lowercase for case-insensitive matching
        text = chunk.page_content.lower()
        
        # Assign a section based on keywords found in the chunk
        if "coverage" in text or "covered" in text:
            section = "coverage"
        elif any(term in text for term in ["exclusion", "excluded", "not covered"]):
            section = "exclusions"
        elif any(term in text for term in ["term", "definition", "meaning"]):
            section = "general"
        else:
            section = "other"

        chunk.metadata["section"] = section
    
    return chunks

def get_gemini_embeddings():
    """
    Initializes and returns GoogleGenerativeAIEmbeddings.
    Ensure GOOGLE_API_KEY is set in your environment or .env file.
    """
    # Use 'models/embedding-001' or other suitable embedding model for Gemini
    # The API key will be picked up automatically from GOOGLE_API_KEY environment variable by langchain-google-genai
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def build_qa_chain(vectorstore):
    """
    Builds a RetrievalQA chain using Gemini LLM and a MultiQueryRetriever.
    """
    # Initialize ChatGoogleGenerativeAI for the LLM
    # Use 'gemini-pro' for conversational tasks.
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    print("Initialized ChatGoogleGenerativeAI (gemini-pro) for QA chain.")

    # Base retriever for similarity search
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5} # Retrieve top 5 most similar documents
    )
    print("Initialized base retriever with similarity search.")

    # MultiQueryRetriever generates multiple queries from a single input query
    # to improve retrieval results.
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )
    print("Initialized MultiQueryRetriever.")

    # Prompt template for the QA chain
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant. Use only the following context to answer the question. If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
"""
    )
    print("Defined prompt template for QA chain.")

    # Build the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True, # Return the source chunks that were used to generate the answer
        chain_type="stuff", # Stuffs all retrieved documents into the prompt
        chain_type_kwargs={"prompt": prompt_template}
    )
    print("Built RetrievalQA chain.")

    return qa_chain

