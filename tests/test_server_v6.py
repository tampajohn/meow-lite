"""Tests for the v5/v6 engines wired into the server (MEOW_LITE_ENGINE selector)."""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from meow_lite import neural_v6, server

REPO_ROOT = Path(__file__).resolve().parent.parent
V5_DIR = REPO_ROOT / "models" / "meow-lite-v5"
V6_DIR = REPO_ROOT / "models" / "meow-lite-v6"

pytestmark = pytest.mark.skipif(
    not (V5_DIR.is_dir() and V6_DIR.is_dir()),
    reason="models/meow-lite-v5 and models/meow-lite-v6 checkpoints not present",
)


@pytest.fixture(autouse=True)
def _restore_v4_default():
    """Every test in this module leaves the server back on the v4 default."""
    yield
    os.environ.pop("MEOW_LITE_ENGINE", None)
    server.reset_engine()


def _load_checkpoint(path: Path):
    engine = neural_v6.load(str(path))
    assert engine is not None, f"checkpoint at {path} must load"
    return engine


@pytest.fixture(scope="module")
def v5_engine():
    """Load the chaotic-cat checkpoint ONCE per session (27MB)."""
    return _load_checkpoint(V5_DIR)


@pytest.fixture(scope="module")
def v6_engine():
    """Load the calmer-cat checkpoint ONCE per session (27MB)."""
    return _load_checkpoint(V6_DIR)


def _switch_engine(monkeypatch, name, engine=None):
    monkeypatch.setenv("MEOW_LITE_ENGINE", name)
    server.reset_engine()
    if engine is not None:
        # inject the session-loaded engine so only one load happens per run
        server._neural_engines[name] = engine
        server._neural_checked.add(name)


def test_health_reports_engine_default_v4():
    server.reset_engine()
    client = TestClient(server.app)
    assert client.get("/health").json() == {"status": "ok", "engine": "v4"}


def test_health_reports_engine_v5(monkeypatch):
    _switch_engine(monkeypatch, "v5")
    client = TestClient(server.app)
    assert client.get("/health").json() == {"status": "ok", "engine": "v5"}


def test_health_reports_engine_v6(monkeypatch):
    _switch_engine(monkeypatch, "v6")
    client = TestClient(server.app)
    assert client.get("/health").json() == {"status": "ok", "engine": "v6"}


def test_v5_chat_completion(monkeypatch, v5_engine):
    _switch_engine(monkeypatch, "v5", v5_engine)
    client = TestClient(server.app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "meow-lite",
            "messages": [{"role": "user", "content": "who's a good cat?"}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["finish_reason"] == "stop"
    content = body["choices"][0]["message"]["content"]
    assert content, "v5 must return a non-empty response"
    usage = body["usage"]
    assert usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"]


def test_v6_chat_completion(monkeypatch, v6_engine):
    _switch_engine(monkeypatch, "v6", v6_engine)
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


def test_v5_path_does_not_consult_behavior(monkeypatch, v5_engine):
    _switch_engine(monkeypatch, "v5", v5_engine)

    def _must_not_run(*args, **kwargs):
        raise AssertionError("v5 path must not consult behavior.py")

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


def test_v6_path_does_not_consult_behavior(monkeypatch, v6_engine):
    _switch_engine(monkeypatch, "v6", v6_engine)

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
    _switch_engine(monkeypatch, "v6", v6_engine)
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
    _switch_engine(monkeypatch, "v6", v6_engine)
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


def test_v5_falls_back_to_v4_on_load_failure(monkeypatch):
    monkeypatch.setenv("MEOW_LITE_ENGINE", "v5")
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


def test_v4_default_unaffected_by_neural_engines(monkeypatch, v5_engine):
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


def test_unknown_engine_falls_back_to_v4(monkeypatch):
    monkeypatch.setenv("MEOW_LITE_ENGINE", "banana")
    server.reset_engine()
    client = TestClient(server.app)
    assert client.get("/health").json() == {"status": "ok", "engine": "v4"}
