"""Gemini Interactions API adapters used by the MAS and belief pipeline.

This module is intentionally separate from ``gpt_request.py`` so the original
Code2Video agent keeps its existing API behaviour.
"""

from __future__ import annotations

import base64
import inspect
import json
import mimetypes
import os
import types
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Union, get_args, get_origin, get_type_hints


_DEFAULT_CLIENT: Any = None
_DEFAULT_CLIENT_LOCK = Lock()
_EVAL_CLIENT: Any = None
_EVAL_CLIENT_LOCK = Lock()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_safe(model_dump())
    return str(value)


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _annotation_schema(annotation: Any) -> Dict[str, Any]:
    if annotation in (inspect.Parameter.empty, Any):
        return {}

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, types.UnionType):
        non_none = [item for item in args if item is not type(None)]
        if len(non_none) == 1:
            return _annotation_schema(non_none[0])
        return {"anyOf": [_annotation_schema(item) for item in non_none]}
    if origin in (list, List, Iterable):
        item_type = args[0] if args else Any
        return {"type": "array", "items": _annotation_schema(item_type)}
    if origin in (dict, Dict):
        return {"type": "object"}
    if annotation is str:
        return {"type": "string"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    return {}


def callable_to_interaction_tool(fn: Callable[..., Any]) -> Dict[str, Any]:
    """Convert an annotated Python callable into an Interactions function tool."""
    signature = inspect.signature(fn)
    try:
        type_hints = get_type_hints(fn)
    except Exception:
        type_hints = {}
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        properties[name] = _annotation_schema(type_hints.get(name, parameter.annotation))
        if parameter.default is inspect.Parameter.empty:
            required.append(name)

    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        parameters["required"] = required

    return {
        "type": "function",
        "name": fn.__name__,
        "description": (inspect.getdoc(fn) or f"Call {fn.__name__}.").strip(),
        "parameters": parameters,
    }


def _usage_value(usage: Any, *names: str) -> int:
    for name in names:
        value = _get(usage, name)
        if value is not None:
            return int(value or 0)
    return 0


class InteractionResponse:
    """Compatibility view over one or more Gemini Interaction resources."""

    def __init__(self, interactions: List[Any]):
        if not interactions:
            raise ValueError("At least one interaction is required.")
        self.interactions = interactions
        self.interaction = interactions[-1]
        self.id = _get(self.interaction, "id")
        self.model = _get(self.interaction, "model")
        self.steps = [
            step
            for interaction in interactions
            for step in (_get(interaction, "steps", []) or [])
        ]
        self.output_text = str(_get(self.interaction, "output_text", "") or "")
        self.text = self.output_text

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        for interaction in interactions:
            usage = _get(interaction, "usage")
            prompt_tokens += _usage_value(
                usage,
                "total_input_tokens",
                "input_tokens",
                "prompt_tokens",
            )
            completion_tokens += _usage_value(
                usage,
                "total_output_tokens",
                "output_tokens",
                "completion_tokens",
            )
            total_tokens += _usage_value(usage, "total_tokens")
        if not total_tokens:
            total_tokens = prompt_tokens + completion_tokens

        self.usage_metadata = SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=completion_tokens,
            total_token_count=total_tokens,
        )
        self.usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        # Compatibility for existing Code2Video parsing helpers. Tool traces use
        # the native ``steps`` property instead.
        part = SimpleNamespace(
            text=self.output_text,
            function_call=None,
            function_response=None,
        )
        self.candidates = [
            SimpleNamespace(content=SimpleNamespace(parts=[part]))
        ]


def _normalise_text_input(input_value: Any) -> Any:
    if isinstance(input_value, str):
        return input_value
    if isinstance(input_value, (list, tuple)):
        if len(input_value) == 1 and isinstance(input_value[0], str):
            return input_value[0]
        if all(isinstance(item, str) for item in input_value):
            return "\n".join(input_value)
    return input_value


def _generation_config(max_output_tokens: Optional[int]) -> Optional[Dict[str, Any]]:
    if max_output_tokens is None:
        return None
    return {"max_output_tokens": int(max_output_tokens)}


def create_interaction(
    client: Any,
    *,
    model: str,
    input_value: Any,
    max_output_tokens: Optional[int] = None,
    response_schema: Any = None,
    store: bool = True,
) -> InteractionResponse:
    kwargs: Dict[str, Any] = {
        "model": model,
        "input": _normalise_text_input(input_value),
        "store": store,
    }
    generation_config = _generation_config(max_output_tokens)
    if generation_config:
        kwargs["generation_config"] = generation_config
    if response_schema is not None:
        schema = (
            response_schema.model_json_schema()
            if hasattr(response_schema, "model_json_schema")
            else response_schema
        )
        kwargs["response_format"] = {
            "type": "text",
            "mime_type": "application/json",
            "schema": schema,
        }
    return InteractionResponse([client.interactions.create(**kwargs)])


def run_tool_interaction(
    client: Any,
    *,
    model: str,
    input_value: Any,
    tools: List[Callable[..., Any]],
    max_remote_calls: int,
    max_output_tokens: Optional[int] = None,
) -> InteractionResponse:
    """Run the explicit Interactions function-call/result loop."""
    tool_map = {tool.__name__: tool for tool in tools}
    declarations = [callable_to_interaction_tool(tool) for tool in tools]
    kwargs: Dict[str, Any] = {
        "model": model,
        "input": _normalise_text_input(input_value),
        "tools": declarations,
        "store": True,
    }
    generation_config = _generation_config(max_output_tokens)
    if generation_config:
        kwargs["generation_config"] = generation_config

    current = client.interactions.create(**kwargs)
    interactions = [current]
    remote_calls = 0

    while True:
        function_calls = [
            step
            for step in (_get(current, "steps", []) or [])
            if _get(step, "type") == "function_call"
        ]
        if not function_calls:
            break

        if remote_calls + len(function_calls) > max_remote_calls:
            requested_names = [str(_get(call, "name") or "") for call in function_calls]
            raise RuntimeError(
                "Gemini interaction function-call budget exceeded before executing "
                f"the next batch: used={remote_calls}, requested={len(function_calls)}, "
                f"limit={max_remote_calls}, functions={requested_names}."
            )

        results: List[Dict[str, Any]] = []
        for call in function_calls:
            remote_calls += 1

            name = str(_get(call, "name") or "")
            arguments = _get(call, "arguments", {}) or {}
            fn = tool_map.get(name)
            if fn is None:
                result: Any = {
                    "ok": False,
                    "error": f"Unknown function requested: {name}",
                }
            else:
                try:
                    result = fn(**dict(arguments))
                    if result is None:
                        result = {"ok": True}
                except Exception as exc:
                    result = {
                        "ok": False,
                        "error": f"{exc.__class__.__name__}: {exc}",
                    }

            results.append(
                {
                    "type": "function_result",
                    "name": name,
                    "call_id": _get(call, "id"),
                    "result": [
                        {
                            "type": "text",
                            "text": json.dumps(_json_safe(result), ensure_ascii=False),
                        }
                    ],
                }
            )

        current = client.interactions.create(
            model=model,
            previous_interaction_id=_get(current, "id"),
            input=results,
            tools=declarations,
            store=True,
            **({"generation_config": generation_config} if generation_config else {}),
        )
        interactions.append(current)

    return InteractionResponse(interactions)


_FILE_CACHE: Dict[tuple, Any] = {}
_FILE_CACHE_LOCK = Lock()


def _mime_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def _uploaded_media_block(client: Any, path: Path, media_type: str) -> Dict[str, Any]:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    with _FILE_CACHE_LOCK:
        uploaded = _FILE_CACHE.get(key)
    if uploaded is None:
        uploaded = client.files.upload(file=str(path))
        while True:
            state = _get(uploaded, "state")
            state_name = str(_get(state, "name", state) or "").upper()
            if state_name in ("", "ACTIVE"):
                break
            if state_name == "FAILED":
                raise RuntimeError(f"Gemini file processing failed for {path}.")
            time.sleep(2)
            uploaded = client.files.get(name=_get(uploaded, "name"))
        with _FILE_CACHE_LOCK:
            _FILE_CACHE[key] = uploaded
    return {
        "type": media_type,
        "uri": _get(uploaded, "uri"),
        "mime_type": _get(uploaded, "mime_type", _mime_type(path)),
    }


def _inline_media_block(path: Path, media_type: str) -> Dict[str, Any]:
    return {
        "type": media_type,
        "data": base64.b64encode(path.read_bytes()).decode("ascii"),
        "mime_type": _mime_type(path),
    }


def media_block(client: Any, path_value: Union[str, Path], media_type: str) -> Dict[str, Any]:
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    # Upload videos for reuse across evaluator questions. Images are small and
    # cheaper to send inline.
    if media_type == "video":
        return _uploaded_media_block(client, path, media_type)
    return _inline_media_block(path, media_type)


def create_multimodal_interaction(
    client: Any,
    *,
    model: str,
    prompt: str,
    video_path: Optional[Union[str, Path]] = None,
    image_path: Optional[Union[str, Path]] = None,
    max_output_tokens: Optional[int] = None,
) -> InteractionResponse:
    inputs: List[Dict[str, Any]] = []
    if video_path is not None:
        inputs.append(media_block(client, video_path, "video"))
    if image_path is not None:
        inputs.append(media_block(client, image_path, "image"))
    inputs.append({"type": "text", "text": prompt})
    return create_interaction(
        client,
        model=model,
        input_value=inputs,
        max_output_tokens=max_output_tokens,
    )


def response_usage_dict(response: InteractionResponse) -> Dict[str, int]:
    usage = response.usage_metadata
    return {
        "prompt_tokens": int(usage.prompt_token_count or 0),
        "completion_tokens": int(usage.candidates_token_count or 0),
        "total_tokens": int(usage.total_token_count or 0),
    }


def _load_api_key(*, evaluation: bool = False) -> str:
    if evaluation:
        key = os.getenv("EVAL_GEMINI_API_KEY")
        if key:
            return key
        config_path = Path(__file__).with_name("api_config.json")
        if config_path.exists():
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            key = str((payload.get("gemini") or {}).get("eval_api_key") or "")
            if key:
                return key

    key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
    if key:
        return key
    config_path = Path(__file__).with_name("api_config.json")
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        key = str((payload.get("gemini") or {}).get("api_key") or "")
    if not key:
        raise ValueError("Missing GEMINI_API_KEY/API_KEY or gemini.api_key.")
    return key


def default_client() -> Any:
    global _DEFAULT_CLIENT
    with _DEFAULT_CLIENT_LOCK:
        if _DEFAULT_CLIENT is None:
            from google import genai

            _DEFAULT_CLIENT = genai.Client(api_key=_load_api_key())
    return _DEFAULT_CLIENT


def evaluation_client() -> Any:
    """Return a client using the evaluation-only key when configured."""
    global _EVAL_CLIENT
    with _EVAL_CLIENT_LOCK:
        if _EVAL_CLIENT is None:
            from google import genai

            _EVAL_CLIENT = genai.Client(api_key=_load_api_key(evaluation=True))
        return _EVAL_CLIENT


def request_interaction_text(
    prompt: str,
    log_id: Optional[str] = None,
    max_tokens: int = 8000,
    max_retries: int = 3,
    model_name: Optional[str] = None,
    use_eval_credentials: bool = False,
) -> InteractionResponse:
    del log_id
    model = model_name or os.getenv("MAS_MODEL", "gemini-3-flash-preview")
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return create_interaction(
                evaluation_client() if use_eval_credentials else default_client(),
                model=model,
                input_value=prompt,
                max_output_tokens=max_tokens,
            )
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                raise
            time.sleep(min(15, 2 ** attempt))
    raise RuntimeError(str(last_error))


def request_interaction_video(
    prompt: str,
    video_path: Union[str, Path],
    log_id: Optional[str] = None,
    max_tokens: int = 8000,
    max_retries: int = 3,
    model_name: Optional[str] = None,
    use_eval_credentials: bool = False,
) -> InteractionResponse:
    del log_id
    model = model_name or os.getenv("EVAL_MODEL", "gemini-2.5-pro")
    for attempt in range(max_retries + 1):
        try:
            return create_multimodal_interaction(
                evaluation_client() if use_eval_credentials else default_client(),
                model=model,
                prompt=prompt,
                video_path=video_path,
                max_output_tokens=max_tokens,
            )
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(min(15, 2 ** attempt))
    raise RuntimeError("Unreachable")


def request_interaction_video_image(
    prompt: str,
    video_path: Union[str, Path],
    image_path: Union[str, Path],
    log_id: Optional[str] = None,
    max_tokens: int = 8000,
    max_retries: int = 3,
    model_name: Optional[str] = None,
) -> tuple[InteractionResponse, Dict[str, int]]:
    del log_id
    model = model_name or os.getenv("MAS_MODEL", "gemini-3-flash-preview")
    for attempt in range(max_retries + 1):
        try:
            response = create_multimodal_interaction(
                default_client(),
                model=model,
                prompt=prompt,
                video_path=video_path,
                image_path=image_path,
                max_output_tokens=max_tokens,
            )
            return response, response_usage_dict(response)
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(min(15, 2 ** attempt))
    raise RuntimeError("Unreachable")
