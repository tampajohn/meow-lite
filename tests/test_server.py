"""Tests for the meow-lite server."""

import re

from fastapi.testclient import TestClient

from meow_lite.server import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_models_contains_meow_lite():
    response = client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    ids = [model["id"] for model in body["data"]]
    assert "meow-lite" in ids
    model = body["data"][0]
    assert model["object"] == "model"
    assert model["owned_by"] == "cat"


def _chat_payload():
    return {
        "model": "meow-lite",
        "messages": [{"role": "user", "content": "Hello, what is the meaning of life?"}],
    }


def test_chat_completion_shape():
    response = client.post("/v1/chat/completions", json=_chat_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "meow-lite"
    assert body["id"].startswith("chatcmpl-meow-")
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["finish_reason"] == "stop"
    text = choice["message"]["content"]
    assert text and "m" in text.lower()
    usage = body["usage"]
    assert usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"]
    assert usage["prompt_tokens"] == len(_chat_payload()["messages"][0]["content"].split())


def test_chat_completion_determinism():
    first = client.post("/v1/chat/completions", json=_chat_payload()).json()
    second = client.post("/v1/chat/completions", json=_chat_payload()).json()
    assert first["choices"][0]["message"]["content"] == second["choices"][0]["message"]["content"]
    assert re.search(r"^[A-Z][a-z ]+[.!?]$", first["choices"][0]["message"]["content"])


def test_different_prompts_differ():
    a = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "tell me about dogs"}]},
    ).json()
    b = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "tell me about fish tanks"}]},
    ).json()
    assert a["choices"][0]["message"]["content"] != b["choices"][0]["message"]["content"]


def test_chat_completion_stream():
    response = client.post("/v1/chat/completions", json={**_chat_payload(), "stream": True})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "data: [DONE]" in body
    assert '"role"' in body
    assert '"finish_reason": "stop"' in body


def test_anthropic_messages_shape():
    response = client.post(
        "/v1/messages",
        json={
            "model": "meow-lite",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Explain quantum computing"}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["id"].startswith("msg_meow_")
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == "meow-lite"
    assert body["content"][0]["type"] == "text"
    assert body["content"][0]["text"]
    assert body["stop_reason"] == "end_turn"
    usage = body["usage"]
    assert usage["input_tokens"] >= 1
    assert usage["output_tokens"] >= 1
