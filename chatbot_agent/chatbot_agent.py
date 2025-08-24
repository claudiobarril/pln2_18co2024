import streamlit as st
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_groq import ChatGroq
from agent import Agent
from tools import search_cv_lara, search_cv_victoria, search_cv_claudio


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

system_prompt = """
Eres un asistente experto en responder preguntas sobre los CVs cargados.
Usa el contexto proporcionado para dar una respuesta clara, asertiva y precisa.
Si la pregunta involucra a más de una persona, combina la información de todos los CVs relevantes.
Evita frases como "según el CV". 
Si no hay información suficiente, dilo directamente.
"""

def main():
    if not GROQ_API_KEY:
        st.error("⚠️ GROQ_API_KEY no está configurada en las variables de entorno")
        st.stop()

    st.title("🤖 Chatbot Agent - NLP II")
    st.markdown("**¡Bienvenido a mi chatbot!**")

    st.sidebar.title('⚙️ Configuración del Chatbot')
    st.sidebar.markdown("---")

    model = st.sidebar.selectbox(
        'Elige un modelo:',
        ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
    )

    conversational_memory_length = st.sidebar.slider(
        'Longitud de la memoria conversacional:',
        min_value=1,
        max_value=10,
        value=5,
    )

    try:
        groq_chat = ChatGroq(
            api_key=GROQ_API_KEY,
            model=model,
            temperature=0.7,
            max_tokens=1000,
        )
    except Exception as e:
        st.sidebar.error(f"❌ Error al conectar con Groq: {str(e)}")
        st.stop()

    agent = Agent(
        model=groq_chat,
        tools=[search_cv_lara, search_cv_victoria, search_cv_claudio],
        system=system_prompt,
        memory_k=conversational_memory_length
    )

    st.markdown("### 💬 Haz una pregunta sobre los integrantes del grupo:")
    user_question = st.text_input(
        "Escribe tu mensaje aquí:",
        placeholder="Por ejemplo: ¿Estudiaron Victoria y Claudio en la misma universidad?",
        label_visibility="collapsed"
    )

    if user_question and user_question.strip():
        with st.spinner('🤔 El agente está pensando...'):
            try:
                response = agent.run(user_question)

                st.markdown("### 🤖 Respuesta:")
                st.markdown(f"""
                <div style="
                    background-color: #1f1f1f;
                    color: #f0f0f0;
                    padding: 15px;
                    border-radius: 10px;
                    border-left: 4px solid #ff7f0e;
                ">
                    {response}
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error al procesar la pregunta: {str(e)}")


if __name__ == "__main__":
    main()
