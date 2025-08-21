# Entrega de trabajos prácticos de NLP II

Alumno: **Claudio Barril** (a1708)  
  ✉️ [claudiobarril@gmail.com](mailto:claudiobarril@gmail.com)

Aquí tienes un ejemplo de documentación para el **README** de tu proyecto en formato **Markdown**, integrando lo que mencionaste (approach inicial, mejoras, MoE, Expert y generateV2):

## TP1: TinyGPT con Mixture of Experts y Estrategias de Sampling

### Descripción
Este proyecto implementa un modelo de lenguaje basado en **GPT** con incorporación de **Mixture of Experts (MoE)**.  
Se busca explorar mejoras en generación de texto mediante distintas estrategias de sampling y la optimización de capas expertas para una mayor capacidad expresiva del modelo.

### Estructura del Proyecto

El contenido del trabajo se encuentra organizado en la carpeta notebooks.
- `checkpoints/`: contiene los modelos guardados entrenados.
- `tp1.ipynb`: Notebook con el desarrollo del trabajo, con los análisis y conclusiones parciales y totales.
- `trainer.py`: Implementación del loop de entrenamiento provisto por la materia.

---

## TP2: Chatbot con RAG y Vector DB

### Descripción
Este proyecto implementa un chatbot utilizando **RAG (Retrieval-Augmented Generation)** y una base de datos vectorial para mejorar la precisión y relevancia de las respuestas generadas.

### Estructura del Proyecto

El contenido del trabajo se encuentra organizado en la carpeta chatbot.
- `cv/`: Carpeta donde debe colocarse el CV a cargar en la base de datos vectorial.
- `cv_loader.py`: Script para cargar el CV en la base de datos vectorial.
- `test_index.py`: Script para probar la indexación del CV en la base de datos vectorial.
- `chatbot.py`: Implementación del chatbot con RAG.

### Decisiones de Diseño

#### Modelo de Embeddings
- Se utiliza `intfloat/multilingual-e5-base` por su capacidad multilingüe, ideal para procesar el CV en español
- El modelo está optimizado para búsqueda semántica y es compatible con consultas en múltiples idiomas
- Genera vectores de 768 dimensiones que capturan efectivamente el significado semántico del texto

#### Procesamiento del CV
- Tamaño de chunk: 800 caracteres
  - Equilibrio entre granularidad y contexto
  - Permite mantener secciones cohesivas del CV como experiencia laboral o educación
- Overlap de 100 caracteres
  - Evita pérdida de contexto entre chunks
  - Mejora la recuperación de información que podría quedar dividida

#### Interfaz del Chatbot
Se mantuvieron las funcionalidades de configuración del curso base:
- Selección de modelos LLM (Llama 3, Mixtral, Gemma)
- Control de memoria conversacional
- Gestión de historial
- Estas opciones permiten experimentar y comparar diferentes configuraciones

### Arquitectura
El chatbot utiliza una arquitectura RAG (Recuperación Aumentada con Generación):
1. Los documentos del CV se almacenan como embeddings en la base de datos vectorial Pinecone
2. Las preguntas del usuario se convierten en embeddings y se comparan con el contenido del CV
3. Las secciones relevantes del CV se recuperan y envían como contexto a un LLM de Groq (Llama 3, Mixtral o Gemma)
4. El LLM genera una respuesta en lenguaje natural basada en el contexto recuperado

### Ejecución

#### Prerequisitos
Configurar variables de entorno en el sistema o archivo `.env`:
```bash
    export PINECONE_API_KEY="tu-clave-pinecone"
    export GROQ_API_KEY="tu-clave-groq"
```

#### Cargar el CV
```bash
    python chatbot/cv_loader.py
```

#### Probar la Indexación
```bash
    python chatbot/test_index.py "{pregunta de ejemplo}"
```

#### Iniciar el Chatbot
1. Ejecutar:
```bash
    streamlit run chatbot/chatbot.py
```
2. Acceder al chatbot en http://localhost:8501

#### Características:

- Elección entre diferentes modelos LLM
- Ajuste de la longitud de memoria de conversación
- Limpieza del historial de conversación
- Interfaz conversacional natural
- Recuperación de información del CV en tiempo real
