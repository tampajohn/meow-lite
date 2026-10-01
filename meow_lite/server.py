"""FastAPI server speaking OpenAI and Anthropic wire formats, in cat."""

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from meow_lite import neural, neural_v6
from meow_lite.behavior import apply_triggers, is_reward, weave
from meow_lite.meow import MeowGenerator

MODEL_ID = "meow-lite"
V6_MODEL_DIR = str(Path(__file__).resolve().parent.parent / "models" / "meow-lite-v6-balanced")

logger = logging.getLogger("meow_lite.server")

app = FastAPI(title="meow-lite", version="1.0.0")
_generator = MeowGenerator()
_engine: Optional[dict] = None
_engine_checked = False
# v6 engine (specs/v6.md rollout): selected via MEOW_LITE_ENGINE, default v4.
_engine_name: str = os.environ.get("MEOW_LITE_ENGINE", "v4")
_v6_engine: Optional[dict] = None
_v6_checked = False


def reset_engine() -> None:
    """Forget loaded models and re-read the engine env (test/startup boundary)."""
    global _engine, _engine_checked, _engine_name, _v6_engine, _v6_checked
    _engine = None
    _engine_checked = False
    _engine_name = os.environ.get("MEOW_LITE_ENGINE", "v4")
    _v6_engine = None
    _v6_checked = False


def _get_engine() -> Optional[dict]:
    global _engine, _engine_checked
    if not _engine_checked:
        _engine_checked = True
        _engine = neural.try_load()
    return _engine


def _get_v6_engine() -> Optional[dict]:
    global _v6_engine, _v6_checked
    if not _v6_checked:
        _v6_checked = True
        _v6_engine = neural_v6.load(V6_MODEL_DIR)
        if _v6_engine is None:
            logger.warning(
                "MEOW_LITE_ENGINE=v6 but the checkpoint at %s failed to load; "
                "falling back to v4 behavior",
                V6_MODEL_DIR,
            )
    return _v6_engine


def _generate(prompt: str, warm: bool = False) -> str:
    engine = _get_engine()
    if engine is not None:
        text = neural.seeded_generate(prompt, engine, warm=warm)
        if text:
            return text
    return _generator.generate(prompt, warm=warm)


def _compose(prompt: str) -> str:
    """Generated meows with forced behavior tokens woven in front.

    v6 engine (server-silent comprehension): the model's response is used
    DIRECTLY — no behavior.py weave, no regex triggers, no reward warming.
    If the v6 checkpoint fails to load, falls back to the v4 behavior path.
    """
    if _engine_name == "v6":
        v6 = _get_v6_engine()
        if v6 is not None:
            return neural_v6.seeded_generate(prompt, v6)

    warm = is_reward(prompt)
    text = weave(apply_triggers(prompt), _generate(prompt, warm=warm))
    # Preserve the v1 contract: output starts with a capital letter.
    if text and text[0].isalpha() and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    stream: bool = False


class AnthropicMessage(BaseModel):
    role: str
    content: Any


class AnthropicMessagesRequest(BaseModel):
    model: Optional[str] = None
    max_tokens: int = 1024
    messages: list[AnthropicMessage]
    stream: bool = False


def _last_user_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            if isinstance(message.content, str):
                return message.content
            if isinstance(message.content, list):
                parts = []
                for block in message.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        parts.append(block)
                return " ".join(parts)
            return str(message.content)
    return ""


def _token_counts(text: str, prompt_text: str) -> tuple[int, int, int]:
    prompt_tokens = len(prompt_text.split())
    completion_tokens = len(text.split())
    return prompt_tokens, completion_tokens, prompt_tokens + completion_tokens


def _sse(data: dict[str, Any]) -> str:
    import json

    return f"data: {json.dumps(data)}\n\n"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "engine": _engine_name}


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "cat",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest) -> Any:
    prompt = _last_user_text(request.messages)
    text = _compose(prompt)
    prompt_tokens, completion_tokens, total_tokens = _token_counts(text, prompt)

    if request.stream:
        words = text.split()

        def stream() -> Any:
            yield _sse(
                {
                    "id": "chatcmpl-meow-" + uuid.uuid4().hex,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": MODEL_ID,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }
                    ],
                }
            )
            for word in words:
                yield _sse(
                    {
                        "id": "chatcmpl-meow-" + uuid.uuid4().hex,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": MODEL_ID,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": word + " "},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            yield _sse(
                {
                    "id": "chatcmpl-meow-" + uuid.uuid4().hex,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": MODEL_ID,
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "stop"}
                    ],
                }
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return {
        "id": "chatcmpl-meow-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


@app.post("/v1/messages")
def anthropic_messages(request: AnthropicMessagesRequest) -> Any:
    prompt = _last_user_text(request.messages)
    text = _compose(prompt)
    _, output_tokens, _ = _token_counts(text, prompt)

    if request.stream:

        def stream() -> Any:
            import json

            def event(name: str, payload: dict[str, Any]) -> str:
                return (
                    f"event: {name}\ndata: {json.dumps(payload)}\n\n"
                )

            message_id = "msg_meow_" + uuid.uuid4().hex
            yield event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "model": MODEL_ID,
                        "content": [],
                        "stop_reason": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                },
            )
            yield event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            for word in text.split():
                yield event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": word + " "},
                    },
                )
            yield event(
                "content_block_stop",
                {"type": "content_block_stop", "index": 0},
            )
            yield event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": output_tokens},
                },
            )
            yield event("message_stop", {"type": "message_stop"})

        return StreamingResponse(stream(), media_type="text/event-stream")

    return {
        "id": "msg_meow_" + uuid.uuid4().hex,
        "type": "message",
        "role": "assistant",
        "model": MODEL_ID,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": len(prompt.split()), "output_tokens": output_tokens},
    }
