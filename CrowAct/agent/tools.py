import importlib.util
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional, get_args, get_origin, get_type_hints

ToolFunc = Callable[..., Any]

_TOOL_FUNCS: dict[str, ToolFunc] = {}
_TOOL_SCHEMAS: list[dict[str, Any]] = []


def _json_schema_type(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin is None:
        if annotation is str:
            return "string"
        if annotation is int:
            return "integer"
        if annotation is float:
            return "number"
        if annotation is bool:
            return "boolean"
        return "string"

    if origin is list:
        return "array"
    if origin is dict:
        return "object"

    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if len(args) == 1:
        return _json_schema_type(args[0])
    return "string"


def _build_input_schema(
    func: ToolFunc, param_descriptions: Optional[dict[str, str]]
) -> dict[str, Any]:
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, parameter in signature.parameters.items():
        annotation = type_hints.get(name, str)
        property_schema: dict[str, Any] = {"type": _json_schema_type(annotation)}
        if param_descriptions and name in param_descriptions:
            property_schema["description"] = param_descriptions[name]
        properties[name] = property_schema
        if parameter.default is inspect.Signature.empty:
            required.append(name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def tool(
    *,
    description: str,
    name: Optional[str] = None,
    param_descriptions: Optional[dict[str, str]] = None,
) -> Callable[[ToolFunc], ToolFunc]:
    def decorator(func: ToolFunc) -> ToolFunc:
        tool_name = name or func.__name__
        _TOOL_FUNCS[tool_name] = func
        _TOOL_SCHEMAS.append(
            {
                "name": tool_name,
                "description": description,
                "input_schema": _build_input_schema(func, param_descriptions),
            }
        )
        return func

    return decorator


def clear_tools() -> None:
    _TOOL_FUNCS.clear()
    _TOOL_SCHEMAS.clear()


def load_tools_from_folder(folder: str) -> None:
    folder_path = Path(folder).expanduser().resolve()
    if not folder_path.is_dir():
        raise RuntimeError(f"Tools folder does not exist: {folder}")

    clear_tools()
    for index, file_path in enumerate(sorted(folder_path.glob("*.py"))):
        module_name = f"_crowact_tool_{folder_path.name}_{index}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load tool file: {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


def get_tools(folder: Optional[str] = None) -> list[dict[str, Any]]:
    if folder is not None:
        load_tools_from_folder(folder)
    return list(_TOOL_SCHEMAS)


def execute_tool_call(tool_call: dict[str, Any]) -> str:
    function_name = tool_call["name"]
    arguments = tool_call.get("input") or {}
    if not isinstance(arguments, dict):
        return "Tool error: arguments must be an object"

    tool_func = _TOOL_FUNCS.get(function_name)
    if tool_func is None:
        return f"Tool error: unknown tool {function_name}"

    try:
        result = tool_func(**arguments)
    except Exception as exc:
        return f"Tool error: {function_name} execution failed: {exc}"
    return str(result)
