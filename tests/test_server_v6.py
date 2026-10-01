"""Tests for the v6 engine wired into the server (MEOW_LITE_ENGINE selector)."""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from meow_lite import neural_v6, server

V6_DIR = Path(__file__).resolve().parent.parent / "models" / "meow-lite-v6-balanced"

pytestmark = pytest.mark.skipif(
    not V6_DIR.is_dir(), reason="models/meow-lite-v6-balanced checkpoint not present"
)


@pytest.fixture(autouse=True)
def _restore_v4_default():
    """Every test in this module leaves the server back on the v4 default."""
    yield
    os.environ.pop("MEOW_LITE_ENGINE", None)
    server.reset_engine()


@pytest.fixture(scope="module")
def v6_engine():
    """Load the real balanced checkpoint ONCE per session (27MB)."""
    engine = neural_v6.load(str(V6_DIR))
    assert engine is not None, "balanced v6 checkpoint must load"
    return engine


def _switch_to_v6(monkeypatch, engine=None):
    monkeypatch.setenv("MEOW_LITE_ENGINE", "v6")
    server.reset_engine()
    if engine is not None:
        # inject the session-loaded engine so only one load happens per run
        server._v6_engine = engine
        server._v6_checked = True


def test_health_reports_engine_default_v4():
    server.reset_engine()
    client = TestClient(server.app)
    assert client.get("/health").json() == {"status": "ok", "engine": "v4"}


def test_health_reports_engine_v6(monkeypatch):
    _switch_to_v6(monkeypatch)
    client = TestClient(server.app)
    assert client.get("/health").json() == {"status": "ok", "engine": "v6"}


def test_v6_chat_completion(monkeypatch, v6_engine):
    _switch_to_v6(monkeypatch, v6_engine)
    client = TestClient(server.app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "meow-lite",
            "messages": [{"role": "user", "content": "can I rub her tummy?"}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["finish_reason"] == "stop"
    content = body["choices"][0]["message"]["content"]
    assert content, "v6 must return a non-empty response"
    # usage accounting unchanged (whitespace counts)
    usage = body["usage"]
    assert (
        usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"]
    )
    assert usage["prompt_tokens"] == len("can I rub her tummy?".split())


def test_v6_path_does_not_consult_behavior(monkeypatch, v6_engine):
    _switch_to_v6(monkeypatch, v6_engine)

    def _must_not_run(*args, **kwargs):
        raise AssertionError("v6 path must not consult behavior.py")

    monkeypatch.setattr(server, "apply_triggers", _must_not_run)
    monkeypatch.setattr(server, "is_reward", _must_not_run)
    monkeypatch.setattr(server, "weave", _must_not_run)

    client = TestClient(server.app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "meow-lite",
            "messages": [{"role": "user", "content": "we go to the vet tomorrow"}],
        },
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"]


def test_v6_chat_completion_stream(monkeypatch, v6_engine):
    _switch_to_v6(monkeypatch, v6_engine)
    client = TestClient(server.app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "meow-lite",
            "messages": [{"role": "user", "content": "who's a good cat?"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert body.rstrip().endswith("data: [DONE]")
    assert '"role"' in body  # role chunk present
    assert "content" in body  # content chunks present


def test_v6_anthropic_stream(monkeypatch, v6_engine):
    _switch_to_v6(monkeypatch, v6_engine)
    client = TestClient(server.app)
    response = client.post(
        "/v1/messages",
        json={
            "model": "meow-lite",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "it is 3am, what now?"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "message_stop" in response.text


def test_v6_falls_back_to_v4_on_load_failure(monkeypatch):
    monkeypatch.setenv("MEOW_LITE_ENGINE", "v6")
    server.reset_engine()
    monkeypatch.setattr(server.neural_v6, "load", lambda path: None)

    client = TestClient(server.app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "meow-lite",
            "messages": [{"role": "user", "content": "can I pet your belly?"}],
        },
    )
    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    # v4 fallback: behavior weave still produces the belly bite
    assert "<bite>" in content


def test_v4_default_unaffected_by_v6_code(monkeypatch, v6_engine):
    client = TestClient(server.app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "meow-lite",
            "messages": [{"role": "user", "content": "can I pet your belly?"}],
        },
    )
    assert response.status_code == 200
    assert "<bite>" in response.json()["choices"][0]["message"]["content"]
    assert client.get("/health").json()["engine"] == "v4"
