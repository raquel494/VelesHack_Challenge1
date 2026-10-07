
import json

from langchain_core.tools import tool

from helpers import (
    ReadFileError,
    ValidateFileError,
    read_file,
    validate_file,
)


# ============================================================
# HERRAMIENTAS
# ============================================================

@tool
def create_folder(path: str) -> str:
    """Create a folder in the IDE workspace. path is relative to the workspace root."""
    return "ok"


@tool
def delete_folder(path: str) -> str:
    """Delete a folder from the IDE workspace."""
    return "ok"


@tool
def create_file(path: str, content: str) -> str:
    """Create a new file in the IDE workspace and open it in the editor.
    path is relative to the workspace root. content is the full text.
    """
    return "ok"


@tool
def edit_file(path: str, content: str) -> str:
    """Replace the WHOLE content of an existing file and open it in the editor.
    content is the complete new text.
    """
    return "ok"


@tool
def delete_file(path: str) -> str:
    """Delete a file from the IDE workspace."""
    return "ok"


@tool
def read_tool_file(path: str) -> str:
    """Read the content of a file in the IDE workspace."""
    return "ok"


@tool
def validate_tool_file(path: str) -> str:
    """Validate a YAML file of the IDE workspace."""
    return "ok"


TOOLS = [
    create_folder,
    delete_folder,
    create_file,
    edit_file,
    delete_file,
    read_tool_file,
    validate_tool_file,
]


# ============================================================
# ARGUMENTOS ESPERADOS
# ============================================================

ACTIONS = {
    "create_folder": ["path"],
    "delete_folder": ["path"],
    "create_file": ["path", "content"],
    "edit_file": ["path", "content"],
    "delete_file": ["path"],
}


MAX_RESULT_CHARS = 6000


# ============================================================
# VALIDACIÓN DE RUTAS
# ============================================================

def _path_ok(path):
    if not isinstance(path, str) or not path.strip():
        return False

    ruta = path.replace("\\", "/")

    return (
        not ruta.startswith("/")
        and ":" not in ruta
        and ".." not in ruta.split("/")
    )


# ============================================================
# EJECUCIÓN DE HERRAMIENTAS
# ============================================================

async def execute_tool(name, args):
    """
    Returns:
        (event for the IDE, text result for the model)
    """

    args = args or {}

    # --------------------------------------------------------
    # CREAR / EDITAR / ELIMINAR
    # --------------------------------------------------------

    if name in ACTIONS:

        faltan = [
            k
            for k in ACTIONS[name]
            if not isinstance(args.get(k), str)
        ]

        if faltan:
            return (
                None,
                f"Error: missing arguments {faltan}.",
            )

        if not _path_ok(args["path"]):
            return (
                None,
                "Error: path must be relative to the "
                "workspace, never absolute and never contain '..'.",
            )

        evento = {
            "action": name,
            **{
                k: args[k]
                for k in ACTIONS[name]
            },
        }

        return (
            evento,
            f"OK: {name} done for {args['path']}.",
        )

    # --------------------------------------------------------
    # LEER ARCHIVO
    # --------------------------------------------------------

    if name == "read_tool_file":

        path = args.get("path", "")

        if not _path_ok(path):
            return (
                None,
                "Error: invalid file path.",
            )

        try:
            contenido = await read_file(path)

        except ReadFileError as exc:
            return (
                None,
                f"Error: {exc}",
            )

        contenido = contenido[:MAX_RESULT_CHARS]

        # Evento para que la interfaz pueda mostrar
        # el contenido del archivo.
        evento = {
            "action": "read_file",
            "path": path,
            "content": contenido,
        }

        return (
            evento,
            contenido,
        )

    # --------------------------------------------------------
    # VALIDAR ARCHIVO
    # --------------------------------------------------------

    if name == "validate_tool_file":

        path = args.get("path", "")

        if not _path_ok(path):
            return (
                None,
                "Error: invalid file path.",
            )

        try:
            informe = await validate_file(path)

        except ValidateFileError as exc:
            return (
                None,
                f"Error: {exc}",
            )

        informe_texto = json.dumps(
            informe,
            ensure_ascii=False,
        )[:MAX_RESULT_CHARS]

        evento = {
            "action": "validate_file",
            "path": path,
            "result": informe,
        }

        return (
            evento,
            informe_texto,
        )

    # --------------------------------------------------------
    # HERRAMIENTA DESCONOCIDA
    # --------------------------------------------------------

    return (
        None,
        f"Error: unknown tool {name}.",
    )
