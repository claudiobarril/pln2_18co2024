import streamlit as st
import os

from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from pinecone import Pinecone


# Configuración de las variables de entorno
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "cv-index"
NAMESPACE = "cv-claudio-barril"

system_prompt = """
Eres un asistente experto en responder preguntas sobre Claudio Barril.
Evita comenzar la respuesta con "Según el CV" o "según el perfil" o similares.
Responde de manera clara y asertiva y precisa usando el contexto que se te proporciona.
Si no sabes la respuesta, di que no tienes información suficiente.
"""

def main():
    # Verificar si la clave API está configurada
    if not GROQ_API_KEY:
        st.error("⚠️ GROQ_API_KEY no está configurada en las variables de entorno")
        st.info("💡 Configura tu clave API: export GROQ_API_KEY='tu-clave-aqui'")
        st.stop()  # Detener la ejecución si no hay clave API

    # Configurar el título y descripción de la aplicación
    st.title("🤖 Chatbot Claudio Barril - NLP II")
    st.markdown("""
    **¡Bienvenido a mi chatbot!** 

    Este chatbot utiliza:
    - 🧠 **Memoria conversacional**: Recuerda el contexto de tu conversación
    - 🔄 **Modelos intercambiables**: Puedes elegir diferentes LLMs
    - 🚀 **Powered by Groq**: Respuestas rápidas y precisas
    """)

    # Barra lateral para configuración del chatbot
    st.sidebar.title('⚙️ Configuración del Chatbot')
    st.sidebar.markdown("---")

    # Selector de modelo LLM disponible en Groq
    st.sidebar.subheader("🧠 Modelo de Lenguaje")
    model = st.sidebar.selectbox(
        'Elige un modelo:',
        [
            'llama3-8b-8192',  # Llama 3 - 8B parámetros, contexto de 8192 tokens
            'mixtral-8x7b-32768',  # Mixtral - Modelo de mezcla de expertos
            'gemma-7b-it'  # Gemma - Modelo de Google optimizado para instrucciones
        ],
        help="Diferentes modelos tienen distintas capacidades y velocidades"
    )

    # Información sobre el modelo seleccionado
    model_info = {
        'llama3-8b-8192': "🦙 Llama 3: Equilibrio entre velocidad y calidad",
        'mixtral-8x7b-32768': "🔀 Mixtral: Modelo de expertos, excelente para tareas complejas",
        'gemma-7b-it': "💎 Gemma: Optimizado para seguir instrucciones"
    }
    st.sidebar.info(model_info.get(model, "Modelo seleccionado"))

    # Control deslizante para la longitud de memoria
    st.sidebar.subheader("🧠 Configuración de Memoria")
    conversational_memory_length = st.sidebar.slider(
        'Longitud de la memoria conversacional:',
        min_value=1,
        max_value=10,
        value=5,
        help="Número de intercambios anteriores que el bot recordará. Más memoria = mayor contexto pero mayor costo computacional"
    )

    # Mostrar información sobre la memoria
    st.sidebar.caption(f"💭 El bot recordará los últimos {conversational_memory_length} intercambios")


    # Configuración de la memoria conversacional
    # Crear objeto de memoria con ventana deslizante
    memory = ConversationBufferWindowMemory(
        k=conversational_memory_length,
        memory_key="historial_chat",
        return_messages=True
    )

    # Inicializar el historial de chat en el estado de la sesión de Streamlit
    # st.session_state permite mantener datos entre ejecuciones de la aplicación
    if 'historial_chat' not in st.session_state:
        st.session_state.historial_chat = []
        st.sidebar.success("💬 Nueva conversación iniciada")
    else:
        # Si ya existe historial, cargarlo en la memoria de LangChain
        for message in st.session_state.historial_chat:
            memory.save_context(
                {'input': message['humano']},
                {'output': message['IA']}
            )

        # Mostrar información del historial en la barra lateral
        st.sidebar.info(f"💬 Conversación con {len(st.session_state.historial_chat)} mensajes")

    # Botón para limpiar el historial
    if st.sidebar.button("🗑️ Limpiar Conversación"):
        st.session_state.historial_chat = []
        st.sidebar.success("✅ Conversación limpiada")
        st.rerun()  # Recargar la aplicación

    # Interfaz principal del chatbot
    # Crear el campo de entrada para las preguntas del usuario
    st.markdown("### 💬 Haz una pregunta sobre mí:")
    user_question = st.text_input(
        "Escribe tu mensaje aquí:",
        placeholder="Por ejemplo: ¿Dónde trabaja Claudio?",
        label_visibility="collapsed"
    )

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    embed_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

    # Inicializar el cliente de ChatGroq con las configuraciones seleccionadas
    try:
        groq_chat = ChatGroq(
            api_key=GROQ_API_KEY,
            model=model,
            temperature=0.7,  # Creatividad de las respuestas (0=determinista, 1=creativo)
            max_tokens=1000,
        )
        st.sidebar.success("✅ Modelo conectado correctamente")
    except Exception as e:
        st.sidebar.error(f"❌ Error al conectar con Groq: {str(e)}")
        st.stop()

    # Procesamiento de la pregunta del usuario
    if user_question and user_question.strip():
        with st.spinner('🤔 El chatbot está pensando...'):

            try:
                # Obtener el contexto del CV desde Pinecone
                vector = embed_model.embed_query(user_question)
                results = index.query(
                    vector=vector,
                    top_k=3,
                    namespace=NAMESPACE,
                    include_metadata=True
                )
                contexto_cv = "\n".join([match["metadata"].get("text", "") for match in results["matches"]])
                prompt = ChatPromptTemplate.from_messages([
                    # Mensaje del sistema - Define el comportamiento del chatbot
                    SystemMessage(
                        content=f"{system_prompt}\nUsa el siguiente contexto del CV para responder:\n{contexto_cv}"),

                    # Historial de conversación
                    MessagesPlaceholder(variable_name="historial_chat"),

                    # Pregunta actual del usuario
                    HumanMessagePromptTemplate.from_template("{human_input}")
                ])


                # LLMChain conecta el modelo de lenguaje con el template y la memoria
                conversation = LLMChain(
                    llm=groq_chat,  # El modelo de lenguaje configurado
                    prompt=prompt,  # El template de conversación
                    verbose=False,  # Desactivar logs detallados para producción
                    memory=memory,  # La memoria conversacional
                )

                # Enviar la pregunta al modelo y obtener la respuesta
                response = conversation.predict(human_input=user_question)

                # Agregar el mensaje al historial de la sesión
                st.session_state.historial_chat.append({'humano': user_question, 'IA': response})

                # Mostrar la respuesta actual destacada
                st.markdown("### 🤖 Respuesta:")
                st.markdown(f"""
                <div style="
                    background-color: #1f1f1f;  /* mismo fondo que la pregunta */
                    color: #f0f0f0;             /* mismo color de texto */
                    padding: 15px;
                    border-radius: 10px;
                    border-left: 4px solid #ff7f0e; /* mismo borde que la pregunta */
                ">
                    {response}
                </div>
                """, unsafe_allow_html=True)

                # Información adicional sobre la respuesta
                st.caption(f"📊 Modelo: {model} | 🧠 Memoria: {conversational_memory_length} mensajes")

            except Exception as e:
                # Manejo de errores durante el procesamiento
                st.error(f"❌ Error al procesar la pregunta: {str(e)}")
                st.info("💡 Verifica tu conexión a internet y la configuración de la API")

if __name__ == "__main__":
    # Punto de entrada de la aplicación
    # Solo ejecutar main() si este archivo se ejecuta directamente
    main()
