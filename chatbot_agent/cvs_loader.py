import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

# Environment configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
CLOUD = os.environ.get("PINECONE_CLOUD") or "aws"
REGION = os.environ.get("PINECONE_REGION") or "us-east-1"
INDEX_NAME = "cvs-index"

# CV configurations
CV_CONFIGS = [
    {"file": "Barril_Claudio_CV.pdf", "namespace": "cv-claudio-barril"},
    {"file": "Teran_Victoria_CV.pdf", "namespace": "cv-victoria-teran"},
    {"file": "Rosenberg_Lara_CV.pdf", "namespace": "cv-lara-rosenberg"},
]


def read_pdf(file_path: str):
    """Load a single PDF document."""
    loader = PyPDFLoader(file_path)
    return loader.load()


def chunk_data(docs, chunk_size=1000, chunk_overlap=100):
    """Split documents into chunks for embeddings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def init_pinecone(index_name: str, dimension: int = 768):
    """Initialize Pinecone connection and index if it doesn't exist."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    spec = ServerlessSpec(cloud=CLOUD, region=REGION)

    if index_name in pc.list_indexes().names():
        print(f"Index '{index_name}' already exists.")
    else:
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            index_name,
            dimension=dimension,
            metric="cosine",
            spec=spec,
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(5)

    return pc


def process_cv(file_path: str, index_name: str, namespace: str, embed_model):
    """Process a single CV file and upload to Pinecone."""
    # 1. Load PDF
    documents = read_pdf(file_path)

    # 2. Chunking
    chunks = chunk_data(documents, chunk_size=800, chunk_overlap=100)
    print(f"Total chunks created for {file_path}: {len(chunks)}")

    # 3. Upsert to Pinecone
    PineconeVectorStore.from_documents(
        documents=chunks,
        index_name=index_name,
        embedding=embed_model,
        namespace=namespace
    )
    print(f"Vectors inserted in index '{index_name}', namespace '{namespace}'")


if __name__ == "__main__":
    # Initialize embeddings model
    embed_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

    # Initialize Pinecone
    pc = init_pinecone(INDEX_NAME, dimension=768)

    # Process each CV
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    cvs_dir = os.path.join(BASE_DIR, "cvs")

    for config in CV_CONFIGS:
        cv_path = os.path.join(cvs_dir, config["file"])
        if os.path.exists(cv_path):
            process_cv(
                cv_path,
                INDEX_NAME,
                config["namespace"],
                embed_model
            )
        else:
            print(f"File not found: {cv_path}")

    # Print final stats
    dense_index = pc.Index(INDEX_NAME)
    stats = dense_index.describe_index_stats()
    print("\nFinal index stats:")
    print(stats["namespaces"])
