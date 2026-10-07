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


load_dotenv()

API_KEY = os.environ.get("API_KEY", "")
BASE_URL = "https://legion1.di.uoa.gr/v1"
MODEL = "llama3.1"

MAX_ROUNDS = 4
MAX_HISTORY = 10
TOOL_NAMES = {t.name for t in TOOLS}

llm = ChatOpenAI(
    model=MODEL,
    base_url=BASE_URL,
    api_key=API_KEY,
    max_completion_tokens=2048,
    temperature=0,
)
llm_tools = llm.bind_tools(TOOLS)

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


# Memoria de las conversaciones
sessions = {}

SYSTEM_PROMPT = """You are Hyperion, the AI assistant inside the HyperAI IDE.

ABOUT HYPER-AI:
HYPER-AI is a research project that aims to revolutionise distributed computing
by integrating IoT, Edge and Cloud (the computing continuum). It creates smart
virtual computing nodes and optimises data-processing applications across a
distributed network.

The HyperAI IDE is the workspace where users build, configure, validate and
deploy applications using YAML app profiles.

YOUR ROLE:
Hyperion is the AI assistant inside the HyperAI IDE: it answers the user's
questions and builds, modifies and manages application files for the user.

You can:

* Answer questions and explain concepts.
* Answer questions about HYPER-AI using the available documentation.
* Help users create and configure applications.
* Read files from the workspace.
* Create new files.
* Modify existing files.
* Delete files when explicitly requested.
* Create and delete folders when explicitly requested.
* Validate files and explain validation errors and warnings.
* Use the available tools whenever an action in the IDE is required.

GENERAL CHAT BEHAVIOUR:

* Behave like a normal helpful AI assistant.
* Understand and use the conversation context.
* Treat short messages as continuations of the current conversation when
  appropriate. For example, if the user is asked for file content and replies
  "hello world", interpret it as the requested file content.
* Answer directly when no IDE action is required.
* Ask a concise clarification question only when the user's intention cannot
  reasonably be determined from the conversation.
* Do not reject a request simply because it does not contain words such as
  "HYPER-AI", "IDE", "file", or "YAML".
* Keep responses concise and useful unless the user asks for more detail.
* Reply in the same language as the user.

TOOL USAGE:

* To perform an action in the IDE, you MUST use the appropriate tool.
* Never claim to have performed an action unless the corresponding tool was
  actually called.
* Never invent tool results, file names, file contents, validation results,
  or completed actions.
* Never write JSON or tool calls directly in the response.

FILE MANAGEMENT:

* All paths must be relative to the workspace root.
* Never use absolute paths or paths containing '..'.
* When creating a file, use create_file and provide the COMPLETE file content.
* If the user provides the content for a new file, create the file directly
  instead of creating an empty file and asking for its content.
* Before modifying an existing file, ALWAYS call read_tool_file first.
* After reading an existing file, use edit_file with the COMPLETE new file
  content. The edit_file tool replaces the entire file.
* Use read_tool_file when the user asks about the contents of a file or when
  you need its contents to perform an operation.
* Only delete files or folders when the user explicitly requests the deletion.
* After performing an IDE action, briefly explain what was done.

VALIDATION:

* When the user asks to validate a file, ALWAYS call validate_tool_file.
* Explain the errors and warnings returned by the validation tool.
* Never say that a file is valid unless validate_tool_file returned
  "valid": true in the current turn.

YAML AND HYPER-AI:

* Follow the YAML examples provided in the HYPER-AI documentation.
* If no suitable YAML example exists, use 'apiVersion: hyper.ai/v1' and tell
  the user to validate the file.
* When answering questions specifically about HYPER-AI, use the provided
  documentation and context.
* If the documentation does not contain the information needed to answer a
  HYPER-AI-specific question, say that the information is not available in
  the documentation. Do not invent technical details.
* HyperAI, Hyper-AI and HYPER-AI refer to the same project.

SAFETY AND ACCURACY:

* Do not invent information about HYPER-AI or the IDE.
* Do not pretend that a tool was used when it was not.
* Do not make destructive changes unless the user explicitly requests them.
* When an operation is ambiguous and could modify or delete user data, ask
  for clarification before acting.

DOCUMENTATION:

{contexto}
"""



def sse(payload):
    return f"data: {json.dumps(payload)}\n\n"


def a_mensajes(history):
    mensajes = []
    for m in history:
        if m["role"] == "user":
            mensajes.append(HumanMessage(content=m["content"]))
        else:
            mensajes.append(AIMessage(content=m["content"]))
    return mensajes


def extraer_llamadas(texto):
    """Find tool calls that the model wrote as plain JSON text."""
    llamadas = []
    decoder = json.JSONDecoder()
    i = 0
    while True:
        i = texto.find("{", i)
        if i == -1:
            break
        try:
            obj, fin = decoder.raw_decode(texto[i:])
        except ValueError:
            i += 1
            continue
        if isinstance(obj, dict) and obj.get("name") in TOOL_NAMES:
            args = obj.get("parameters") or obj.get("arguments") or {}
            if isinstance(args, dict):
                llamadas.append(
                    {"name": obj["name"], "args": args, "id": f"text-{len(llamadas)}"}
                )
        i += fin
    return llamadas


async def generate_reply(request: ChatRequest):
    # --- GUARDRAIL SIMPLE ---
    # Convertimos la pregunta a minúsculas para buscar palabras clave
    pregunta = request.text.lower()
    
    # Lista de palabras permitidas (conceptos clave del proyecto y herramientas)
    palabras_clave = ["hyper-ai", "hyperai", "ide", "connector", "connectors", 
                      "deploy", "deployment", "yaml", "archivo", "carpeta", 
                      "explicar", "explicado", "qué", "crea", "borra", "valida"]
    
    # Si la pregunta no contiene NINGUNA de las palabras clave, la rechazamos
    if not any(palabra in pregunta for palabra in palabras_clave):
        yield sse({"response": "Lo siento, como asistente de HYPER-AI solo puedo responder preguntas relacionadas con el proyecto, el IDE o realizar acciones sobre los archivos."})
        yield "data: [DONE]\n\n"
        return
    # ------------------------

    history = sessions.setdefault(request.user_id, [])

    resultados = buscar_documentacion(request.text, k=3)
    contexto = "\n\n".join(
        f"DOCUMENT: {r['filename']}\n{r['text']}" for r in resultados
    )

    # Actualizamos el system prompt con el nuevo contexto del RAG
    system_msg = SystemMessage(content=SYSTEM_PROMPT.format(contexto=contexto))
    
    # Construimos la lista de mensajes: System + Historial (sin el actual) + User actual
    messages = [system_msg]
    if history:
         messages += a_mensajes(history[-MAX_HISTORY:])
    
    # Añadimos la pregunta actual a la lista de mensajes que enviamos al LLM
    messages.append(HumanMessage(content=request.text))

    full_response = ""

    try:
        for _ in range(MAX_ROUNDS):
            final = None
            ronda = ""
            retenido = ""
            reteniendo = False

            async for chunk in llm_tools.astream(messages):
                final = chunk if final is None else final + chunk
                t = chunk.text
                if not t:
                    continue
                ronda += t
                if reteniendo:
                    retenido += t
                elif "{" in t:
                    antes, _, despues = t.partition("{")
                    if antes:
                        full_response += antes
                        yield sse({"response": antes})
                    reteniendo = True
                    retenido = "{" + despues
                else:
                    full_response += t
                    yield sse({"response": t})

            nativas = bool(final is not None and final.tool_calls)
            if nativas:
                llamadas = list(final.tool_calls)
            elif retenido:
                llamadas = extraer_llamadas(ronda)
            else:
                llamadas = []

            if not llamadas:
                if retenido:
                    full_response += retenido
                    yield sse({"response": retenido})
                break

            if nativas:
                messages.append(final)
            else:
                messages.append(AIMessage(content=ronda))

            for call in llamadas:
                evento, resultado = await execute_tool(call["name"], call["args"])
                if evento:
                    yield sse(evento)
                if nativas:
                    messages.append(
                        ToolMessage(
                            content=resultado,
                            tool_call_id=call.get("id") or call["name"],
                        )
                    )
                else:
                    messages.append(
                        HumanMessage(
                            content=f"Result of {call['name']}: {resultado}"
                        )
                    )
    except Exception as exc:
        yield sse({"response": f"\n\n[Error: {exc}]"})

    history.append({"role": "user", "content": request.text})
    history.append({"role": "assistant", "content": full_response})

    yield "data: [DONE]\n\n"


@app.post("/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(
        generate_reply(request),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)