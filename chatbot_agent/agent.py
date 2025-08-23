import unicodedata
import re

from typing import Dict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory


class Agent:
    person_patterns = {
        "cv-claudio-barril": re.compile(r"\b(claudio|barril)\b", re.IGNORECASE),
        "cv-victoria-teran": re.compile(r"\b(victoria|ter[aá]n)\b", re.IGNORECASE),
        "cv-lara-rosenberg": re.compile(r"\b(lara|rosenberg)\b", re.IGNORECASE),
    }
    default_namespace = "cv-claudio-barril"

    def __init__(self, model, tools, system="", memory_k=5):
        self.system = system
        self.tools = {t.name: t for t in tools}
        self.model = model
        self.memory = ConversationBufferWindowMemory(
            k=memory_k,
            memory_key="chat_history",
            return_messages=True
        )

    @staticmethod
    def normalizar(texto: str) -> str:
        texto = texto.lower()
        texto = unicodedata.normalize('NFD', texto)
        return ''.join(c for c in texto if unicodedata.category(c) != 'Mn')

    def choose_namespaces(self, pregunta: str) -> list:
        nq = self.normalizar(pregunta)
        matches = [ns for ns, pat in self.person_patterns.items() if pat.search(nq)]
        return matches if matches else [self.default_namespace]

    def run(self, user_input: str) -> str:
        # 1. Get relevant namespaces
        namespaces = self.choose_namespaces(user_input)

        # 2. Define namespace to tool mapping
        namespace_to_tool: Dict[str, str] = {
            "cv-claudio-barril": "search_cv_claudio",
            "cv-victoria-teran": "search_cv_victoria",
            "cv-lara-rosenberg": "search_cv_lara"
        }

        # 3. Gather context per student
        student_contexts = {}
        for namespace in namespaces:
            tool_name = namespace_to_tool.get(namespace)
            if tool_name and tool_name in self.tools:
                result = self.tools[tool_name].invoke(user_input)
                student_name = tool_name.replace('search_cv_', '').title()
                student_contexts[student_name] = result

        # 4. Create enhanced prompt with separated contexts
        context_sections = []
        for student, context in student_contexts.items():
            section = f"Información de {student}:\n{context}"
            context_sections.append(section)

        context = "\n\n".join(context_sections)
        enhanced_prompt = (
            f"Pregunta: {user_input}\n\n"
            f"{context}\n\n"
            "Responde la pregunta organizando la información por persona. "
            "Si la información corresponde a varios estudiantes, indica claramente "
            "a quién pertenece cada dato."
        )

        # 5. Generate final response
        messages = []
        if self.system:
            messages.append(SystemMessage(content=self.system))
        history = self.memory.load_memory_variables({}).get("chat_history", [])
        messages.extend(history)
        messages.append(HumanMessage(content=enhanced_prompt))

        response = self.model.invoke(messages)

        # 6. Save to memory and return
        self.memory.save_context(
            {"input": user_input},
            {"output": response.content}
        )
        return response.content
