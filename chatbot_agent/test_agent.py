import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_groq import ChatGroq
from agent import Agent
from tools import search_cv_lara, search_cv_victoria, search_cv_claudio

# Configuración
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

SYSTEM_PROMPT = """
Eres un asistente experto en responder preguntas sobre los CVs cargados.
Usa el contexto proporcionado para dar una respuesta clara, asertiva y precisa.
Si la pregunta involucra a más de una persona, combina la información de todos los CVs relevantes.
Evita frases como "según el CV". 
Si no hay información suficiente, dilo directamente.
"""


def main():
    if len(sys.argv) < 2:
        print("Uso: python test_message.py 'Tu pregunta aquí'")
        print("Ejemplo: python test_message.py '¿Dónde estudió Victoria?'")
        sys.exit(1)

    # Verificar variables de entorno
    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY no está configurada")
        print("Configúrala con: export GROQ_API_KEY='tu-clave-groq'")
        sys.exit(1)

    if not PINECONE_API_KEY:
        print("❌ PINECONE_API_KEY no está configurada")
        print("Configúrala con: export PINECONE_API_KEY='tu-clave-pinecone'")
        sys.exit(1)

    question = sys.argv[1]

    try:
        # Crear modelo LLM
        groq_chat = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama3-8b-8192",
            temperature=0.7,
            max_tokens=1000,
        )

        # Crear agente
        agent = Agent(
            model=groq_chat,
            tools=[search_cv_lara, search_cv_victoria, search_cv_claudio],
            system=SYSTEM_PROMPT,
            memory_k=5
        )

        print("Pregunta:", question)
        print("\n🤔 El agente está procesando...\n")

        # Ejecutar consulta
        response = agent.run(question)

        print("Respuesta del agente:")
        print("-" * 50)
        print(response)
        print("-" * 50)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()