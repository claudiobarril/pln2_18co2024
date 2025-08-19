import os
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

# ======================
# CONFIG
# ======================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "cv-index"
NAMESPACE = "cv-claudio-barril"
CLOUD = os.environ.get("PINECONE_CLOUD") or "aws"
REGION = os.environ.get("PINECONE_REGION") or "us-east-1"

# ======================
# FUNCIONES AUXILIARES
# ======================
def read_pdfs(directory: str):
    """Carga documentos PDF desde un directorio."""
    loader = PyPDFDirectoryLoader(directory)
    return loader.load()

def chunk_data(docs, chunk_size=1000, chunk_overlap=100):
    """Divide documentos en chunks para embeddings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

def init_pinecone(index_name: str, dimension: int = 768):
    """Crea conexión e índice en Pinecone si no existe."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    spec = ServerlessSpec(cloud=CLOUD, region=REGION)

    if index_name in pc.list_indexes().names():
        print(f"El índice '{index_name}' ya existe.")
    else:
        print(f"Creando índice '{index_name}' ...")
        pc.create_index(
            index_name,
            dimension=dimension,
            metric="cosine",
            spec=spec,
        )
        # esperar a que se cree
        time.sleep(3)

    return pc

# ======================
# PIPELINE
# ======================
if __name__ == "__main__":
    # 1. Cargar PDFs
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dir_cv = os.path.join(BASE_DIR, "cv")
    documents = read_pdfs(dir_cv)

    # 2. Chunking
    chunks = chunk_data(documents, chunk_size=800, chunk_overlap=100)
    print(f"Total chunks creados: {len(chunks)}")

    # 3. Inicializar embeddings (HuggingFace wrapper)
    embed_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

    # 4. Pinecone Init
    pc = init_pinecone(INDEX_NAME, dimension=768)

    # 5. Upsert a Pinecone con VectorStore
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        index_name=INDEX_NAME,
        embedding=embed_model,
        namespace=NAMESPACE
    )

    print(f"Vectores insertados en índice '{INDEX_NAME}', namespace '{NAMESPACE}'.")
    dense_index = pc.Index(INDEX_NAME)
    stats = dense_index.describe_index_stats()
    print(stats)
