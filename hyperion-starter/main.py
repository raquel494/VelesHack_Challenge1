
import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from rag import buscar_documentacion
from tools import TOOLS, execute_tool


# ============================================================
# CONFIGURACIÓN
# ============================================================

load_dotenv()

API_KEY = os.environ.get("API_KEY", "")
BASE_URL = "https://legion1.di.uoa.gr/v1"
MODEL = "llama3.1"

MAX_ROUNDS = 4
MAX_HISTORY = 10
RAG_RESULTS = 3

TOOL_NAMES = {tool.name for tool in TOOLS}


# ============================================================
# MODELO
# ============================================================

llm = ChatOpenAI(
    model=MODEL,
    base_url=BASE_URL,
    api_key=API_KEY,
    max_completion_tokens=2048,
    temperature=0,
)

llm_tools = llm.bind_tools(TOOLS)


# ============================================================
# API
# ============================================================

app = FastAPI(title="Hyperion Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    user_id: str
    text: str


# ============================================================
# MEMORIA
# ============================================================

sessions = {}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Hyperion, the AI assistant inside the HyperAI IDE.

ABOUT HYPER-AI:

HYPER-AI is a research project that aims to revolutionise distributed
computing by integrating IoT, Edge and Cloud into the computing continuum.

It creates smart virtual computing nodes and optimises data-processing
applications across a distributed network.

The HyperAI IDE is the workspace where users build, configure, validate
and deploy applications using YAML application profiles.


YOUR ROLE:

Hyperion is the AI assistant inside the HyperAI IDE.

Your job is to help users understand HYPER-AI, work with the IDE and
manage application files through the available tools.

You can:

- Answer questions and explain concepts.
- Answer questions about HYPER-AI using the available documentation.
- Help users create and configure applications.
- Read files from the workspace.
- Create new files.
- Modify existing files.
- Delete files when explicitly requested.
- Create and delete folders when explicitly requested.
- Validate files and explain validation errors and warnings.
- Use the available tools whenever an action in the IDE is required.
- Maintain context from the current conversation.


CONVERSATIONAL BEHAVIOUR:

- Behave naturally and conversationally.
- Respond directly to greetings, thanks, confirmations and short
  conversational messages.
- Do not treat greetings as technical questions.
- Avoid unnecessarily formal, repetitive or robotic responses.
- Keep simple conversational responses short and friendly.
- For technical questions, provide a clear and useful explanation.
- Use the conversation history to understand references such as
  "eso", "esto", "ese archivo", "lo anterior", "hazlo" or "cámbialo".
- If the user's intention is clear, do not ask unnecessary questions.
- If the intention is genuinely ambiguous, ask one concise clarification.
- Reply in the same language as the user.


SCOPE:

- Hyperion is specialised in HYPER-AI and the HyperAI IDE.
- Basic conversation such as greetings, thanks and acknowledgements
  is allowed.
- Continue ongoing HYPER-AI or IDE tasks even when the user's message
  is short.
- Do not provide unrelated general-purpose assistance when it has no
  connection with HYPER-AI, the IDE or the current task.
- For unrelated requests, briefly explain that you are specialised
  in HYPER-AI and the IDE and offer help with those topics instead.


TOOL USAGE:

- If an IDE action is required, ALWAYS use the appropriate tool.
- Never claim that an action was performed unless the tool was actually
  called.
- Never invent tool results, file names, file contents or validation
  results.
- Never write tool calls or JSON directly as the final response.
- If a tool is available for an action, use it instead of merely
  explaining how the user could perform it.
- After using a tool, clearly explain the result to the user.
- Do NOT use file tools for general conceptual questions about HYPER-AI.
- Questions such as "what is HYPER-AI?", "what is HyperAI?",
  "what does HYPER-AI do?" or similar conceptual questions should be
  answered directly using the system context and documentation.
- Only use file tools when the user explicitly asks to read, create,
  modify, validate or delete a workspace file.


FILE MANAGEMENT:

- All paths must be relative to the workspace root.
- Never use absolute paths.
- Never use paths containing '..'.
- When creating a file, use create_file with the COMPLETE file content.
- If the user provides content for a new file, create it directly.
- Before modifying an existing file, ALWAYS call read_tool_file first.
- After reading an existing file, use edit_file with the COMPLETE new
  file content.
- The edit_file tool replaces the entire file.
- Use read_tool_file when the user asks about a file or when its contents
  are required for an operation.
- Only delete files or folders when explicitly requested.
- Never perform destructive operations based only on assumptions.


VALIDATION:

- When the user asks to validate a file, ALWAYS call validate_tool_file.
- Explain the errors and warnings returned by the validation tool.
- Never say that a file is valid unless validate_tool_file returned
  "valid": true in the current turn.


YAML AND HYPER-AI:

- Follow YAML examples provided in the HYPER-AI documentation.
- Prefer documented examples over assumptions.
- If no suitable YAML example exists, use:

  apiVersion: hyper.ai/v1

  and tell the user to validate the file.

- When answering questions specifically about HYPER-AI, use the provided
  documentation and context.
- If the documentation does not contain the required information, say
  that it is not available in the documentation.
- Never invent technical details about HYPER-AI.
- HyperAI, Hyper-AI and HYPER-AI refer to the same project.


SAFETY AND ACCURACY:

- Do not invent information about HYPER-AI or the IDE.
- Do not pretend that a tool was used when it was not.
- Do not make destructive changes without explicit user confirmation.
- When an operation could modify or delete user data and the intention
  is ambiguous, ask for clarification.

CORE KNOWLEDGE:

HYPER-AI is a research project focused on distributed computing.
It integrates IoT, Edge and Cloud resources into the computing continuum.
It creates smart virtual computing nodes and optimises data-processing
applications across a distributed network.

When the user asks a basic conceptual question about HYPER-AI, use this
information directly unless the documentation provides more specific
information.


DOCUMENTATION:

{contexto}
"""


# ============================================================
# RESPUESTAS BÁSICAS
# ============================================================

RESPUESTAS_BASICAS = {
    "hola": "¡Hola! 👋 Soy Hyperion, el asistente de HYPER-AI. ¿En qué puedo ayudarte?",
    "hola!": "¡Hola! 👋 Soy Hyperion, el asistente de HYPER-AI. ¿En qué puedo ayudarte?",
    "hola.": "¡Hola! 👋 Soy Hyperion, el asistente de HYPER-AI. ¿En qué puedo ayudarte?",
    "buenas": "¡Buenas! 👋 ¿Qué necesitas hacer en HYPER-AI o en el IDE?",
    "buenos dias": "¡Buenos días! 👋 ¿En qué puedo ayudarte con HYPER-AI?",
    "buenos días": "¡Buenos días! 👋 ¿En qué puedo ayudarte con HYPER-AI?",
    "buenas tardes": "¡Buenas tardes! 👋 ¿Qué necesitas hacer en el IDE?",
    "buenas noches": "¡Buenas noches! 👋 ¿En qué puedo ayudarte?",
    "gracias": "¡De nada! 😊 Si necesitas algo más con HYPER-AI o el IDE, aquí estoy.",
    "muchas gracias": "¡De nada! 😊 Encantado de ayudar.",
    "adios": "¡Hasta luego! 👋",
    "adiós": "¡Hasta luego! 👋",
    "hasta luego": "¡Hasta luego! 👋",
    "ok": "Perfecto 👍",
    "vale": "Perfecto 👍",
    "perfecto": "¡Genial! 👍",
    "genial": "¡Genial! 👍",
    "bien": "Perfecto 😊",
    "entendido": "Perfecto 👍",
    "de acuerdo": "De acuerdo 👍",
    "qué puedes hacer": (
    "Puedo ayudarte con HYPER-AI y el HyperAI IDE. "
    "Puedo responder preguntas sobre el proyecto y su documentación, "
    "crear, leer, modificar y eliminar archivos, crear carpetas "
    "y validar configuraciones YAML."
),

"que puedes hacer": (
    "Puedo ayudarte con HYPER-AI y el HyperAI IDE. "
    "Puedo responder preguntas sobre el proyecto y su documentación, "
    "crear, leer, modificar y eliminar archivos, crear carpetas "
    "y validar configuraciones YAML."
),
}


# ============================================================
# UTILIDADES
# ============================================================

def sse(payload):
    """Create a Server-Sent Event message."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def a_mensajes(history):
    """Convert stored history into LangChain messages."""

    mensajes = []

    for message in history:

        if message["role"] == "user":
            mensajes.append(
                HumanMessage(
                    content=message["content"]
                )
            )

        elif message["role"] == "assistant":
            mensajes.append(
                AIMessage(
                    content=message["content"]
                )
            )

    return mensajes


def extraer_llamadas(texto):
    """
    Fallback for models that return tool calls as plain JSON.

    Expected format:

    {
        "name": "tool_name",
        "parameters": {...}
    }
    """

    llamadas = []
    decoder = json.JSONDecoder()
    posicion = 0

    while posicion < len(texto):

        inicio = texto.find("{", posicion)

        if inicio == -1:
            break

        try:
            obj, fin = decoder.raw_decode(
                texto[inicio:]
            )

        except ValueError:
            posicion = inicio + 1
            continue

        if (
            isinstance(obj, dict)
            and obj.get("name") in TOOL_NAMES
        ):

            args = (
                obj.get("parameters")
                or obj.get("arguments")
                or {}
            )

            if isinstance(args, dict):

                llamadas.append(
                    {
                        "name": obj["name"],
                        "args": args,
                        "id": f"text-{len(llamadas)}",
                    }
                )

        posicion = inicio + fin

    return llamadas


# ============================================================
# GUARDRAIL
# ============================================================

def mensaje_permitido(texto, history):
    """
    Determines whether the message belongs to Hyperion's scope.
    """

    pregunta = texto.lower().strip()

    if not pregunta:
        return False

    # --------------------------------------------------------
    # Conversación básica
    # --------------------------------------------------------

    if pregunta in RESPUESTAS_BASICAS:
        return True

    # --------------------------------------------------------
    # Palabras relacionadas con HYPER-AI / IDE
    # --------------------------------------------------------

    palabras_clave = [
        "hyper-ai",
        "hyperai",
        "hyper ai",
        "ide",
        "connector",
        "connectors",
        "deploy",
        "deployment",
        "yaml",
        "api",
        "archivo",
        "archivos",
        "fichero",
        "ficheros",
        "carpeta",
        "carpetas",
        "workspace",
        "crear",
        "crea",
        "crear archivo",
        "borrar",
        "borra",
        "eliminar",
        "elimina",
        "editar",
        "edita",
        "modificar",
        "modifica",
        "configurar",
        "configura",
        "validar",
        "valida",
        "validación",
        "documentación",
        "documentacion",
        "herramienta",
        "herramientas",
        "tool",
        "tools",
        "proyecto",
        "aplicación",
        "aplicacion",
    ]

    if any(
        palabra in pregunta
        for palabra in palabras_clave
    ):
        return True

    # --------------------------------------------------------
    # Continuaciones claras
    # --------------------------------------------------------

    if history:

        referencias = [
            "eso",
            "esto",
            "esa",
            "ese",
            "lo anterior",
            "antes",
            "ahora",
            "también",
            "tambien",
            "hazlo",
            "haz eso",
            "continúa",
            "continua",
            "sigue",
            "cámbialo",
            "cambialo",
            "modifícalo",
            "modificalo",
            "corrígelo",
            "corrigelo",
            "elimínalo",
            "eliminalo",
        ]

        if any(
            referencia in pregunta
            for referencia in referencias
        ):
            return True

        # Respuestas cortas a preguntas anteriores.
        respuestas_cortas = {
            "sí",
            "si",
            "no",
            "vale",
            "ok",
            "perfecto",
            "genial",
            "entendido",
            "de acuerdo",
        }

        if pregunta in respuestas_cortas:
            return True

        # Si el último mensaje del asistente estaba hablando
        # claramente del proyecto, permitimos una continuación.
        historial_reciente = history[-4:]

        historial_texto = " ".join(
            message["content"].lower()
            for message in historial_reciente
        )

        if any(
            palabra in historial_texto
            for palabra in palabras_clave
        ):
            # Solo permitimos mensajes relativamente cortos.
            if len(pregunta.split()) <= 12:
                return True

    # --------------------------------------------------------
    # Todo lo demás queda fuera del ámbito de Hyperion
    # --------------------------------------------------------

    return False


# ============================================================
# RAG
# ============================================================

def obtener_contexto(pregunta):
    """Retrieve relevant HYPER-AI documentation."""

    try:

        resultados = buscar_documentacion(
            pregunta,
            k=RAG_RESULTS,
        )

    except Exception:
        return ""

    if not resultados:
        return ""

    documentos = []

    for resultado in resultados:

        documentos.append(
            f"DOCUMENT: {resultado['filename']}\n"
            f"{resultado['text']}"
        )

    return "\n\n".join(documentos)


# ============================================================
# GENERACIÓN DE RESPUESTAS
# ============================================================

async def generate_reply(request: ChatRequest):

    texto_usuario = request.text.strip()

    history = sessions.setdefault(
        request.user_id,
        [],
    )

    # --------------------------------------------------------
    # GUARDRAIL
    # --------------------------------------------------------

    if not mensaje_permitido(
        texto_usuario,
        history,
    ):

        respuesta = (
            "Puedo ayudarte con HYPER-AI, el HyperAI IDE, "
            "sus archivos, configuraciones, herramientas y "
            "tareas relacionadas con el proyecto."
        )

        yield sse(
            {
                "response": respuesta
            }
        )

        yield "data: [DONE]\n\n"
        return

    # --------------------------------------------------------
    # RESPUESTAS BÁSICAS
    # --------------------------------------------------------

    mensaje_basico = texto_usuario.lower()

    if mensaje_basico in RESPUESTAS_BASICAS:

        respuesta = RESPUESTAS_BASICAS[
            mensaje_basico
        ]

        history.append(
            {
                "role": "user",
                "content": texto_usuario,
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": respuesta,
            }
        )

        yield sse(
            {
                "response": respuesta
            }
        )

        yield "data: [DONE]\n\n"
        return

    # --------------------------------------------------------
    # DOCUMENTACIÓN RAG
    # --------------------------------------------------------

    contexto = obtener_contexto(
        texto_usuario
    )

    # --------------------------------------------------------
    # SYSTEM MESSAGE
    # --------------------------------------------------------

    system_msg = SystemMessage(
        content=SYSTEM_PROMPT.format(
            contexto=contexto
        )
    )

    messages = [
        system_msg
    ]

    # Historial reciente.
    if history:

        messages.extend(
            a_mensajes(
                history[-MAX_HISTORY:]
            )
        )

    # Mensaje actual.
    messages.append(
        HumanMessage(
            content=texto_usuario
        )
    )

    full_response = ""

    # ========================================================
    # BUCLE DEL AGENTE
    # ========================================================

    try:

        for _ in range(MAX_ROUNDS):

            final = None
            ronda = ""

            async for chunk in llm_tools.astream(
                messages
            ):

                final = (
                    chunk
                    if final is None
                    else final + chunk
                )

                texto = chunk.text

                if texto:
                    ronda += texto

            # ------------------------------------------------
            # TOOL CALLS NATIVAS
            # ------------------------------------------------

            if (
                final is not None
                and final.tool_calls
            ):

                llamadas = list(
                    final.tool_calls
                )

                messages.append(final)

                for call in llamadas:

                    nombre = call["name"]

                    argumentos = call.get(
                        "args",
                        {},
                    )

                    evento, resultado = (
                        await execute_tool(
                            nombre,
                            argumentos,
                        )
                    )

                    if evento:
                        yield sse(evento)

                    messages.append(
                        ToolMessage(
                            content=str(resultado),
                            tool_call_id=(
                                call.get("id")
                                or nombre
                            ),
                        )
                    )

                continue

            # ------------------------------------------------
            # TOOL CALLS COMO JSON
            # ------------------------------------------------

            llamadas = extraer_llamadas(
                ronda
            )

            if llamadas:

                messages.append(
                    AIMessage(
                        content=ronda
                    )
                )

                for call in llamadas:

                    evento, resultado = (
                        await execute_tool(
                            call["name"],
                            call["args"],
                        )
                    )

                    if evento:
                        yield sse(evento)

                    messages.append(
                        HumanMessage(
                            content=(
                                f"Tool result for "
                                f"{call['name']}:\n"
                                f"{resultado}"
                            )
                        )
                    )

                continue

            # ------------------------------------------------
            # RESPUESTA NORMAL
            # ------------------------------------------------

            if ronda:

                full_response += ronda

                yield sse(
                    {
                        "response": ronda
                    }
                )

            break

    except Exception as exc:

        error = (
            "\n\n[Error interno del asistente: "
            f"{exc}]"
        )

        full_response += error

        yield sse(
            {
                "response": error
            }
        )

    # ========================================================
    # MEMORIA
    # ========================================================

    history.append(
        {
            "role": "user",
            "content": texto_usuario,
        }
    )

    history.append(
        {
            "role": "assistant",
            "content": full_response,
        }
    )

    # Limitar memoria.
    max_messages = MAX_HISTORY * 2

    if len(history) > max_messages:

        del history[
            :-max_messages
        ]

    yield "data: [DONE]\n\n"


# ============================================================
# ENDPOINT
# ============================================================

@app.post("/chat")
async def chat(
    request: ChatRequest,
):

    return StreamingResponse(
        generate_reply(request),
        media_type="text/event-stream",
    )


# ============================================================
# EJECUCIÓN LOCAL
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )

