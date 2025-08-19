import os
import sys
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings

# Configura tus variables de entorno o reemplaza directamente
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = "cv-index"
NAMESPACE = "cv-claudio-barril"

def main():
    if len(sys.argv) < 2:
        print("Uso: python test_index.py 'Tu pregunta aquí'")
        sys.exit(1)

    question = sys.argv[1]

    # Inicializar Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Conectarse al índice
    index = pc.Index(INDEX_NAME)

    # Embeddings
    embed_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    query_vec = embed_model.embed_query(question)

    # Buscar vector más cercano con metadata
    results = index.query(
        vector=query_vec,
        top_k=1,
        namespace=NAMESPACE,
        include_metadata=True
    )

    print("Pregunta:", question)
    print("Resultado más cercano:\n")
    for match in results["matches"]:
        score = match["score"]
        text = match["metadata"].get("text", "")
        print(f"Score: {score:.4f}")
        print(f"Texto: {text}\n")

if __name__ == "__main__":
    main()
