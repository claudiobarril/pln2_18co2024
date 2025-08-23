import os
from langchain_core.tools import tool
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings

INDEX_NAME = "cvs-index"
MODEL_NAME = "intfloat/multilingual-e5-base"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize global dependencies
pc = Pinecone(api_key=PINECONE_API_KEY)
embed_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)

@tool
def search_cv_claudio(query: str) -> str:
    """Search for information in Claudio's CV."""
    return search_in_cv(query, "cv-claudio-barril")

@tool
def search_cv_victoria(query: str) -> str:
    """Search for information in Victoria's CV."""
    return search_in_cv(query, "cv-victoria-teran")

@tool
def search_cv_lara(query: str) -> str:
    """Search for information in Lara's CV."""
    return search_in_cv(query, "cv-lara-rosenberg")

def search_in_cv(question: str, namespace: str) -> str:
    """Search in a specific CV namespace."""
    index = pc.Index(INDEX_NAME)
    vector = embed_model.embed_query(question)

    results = index.query(
        vector=vector,
        top_k=3,
        namespace=namespace,
        include_metadata=True
    )
    return "\n".join(m["metadata"].get("text", "") for m in results["matches"])
