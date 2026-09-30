"""Tests for the v2 tiny neural meow model."""

import pytest
import torch
from fastapi.testclient import TestClient
from transformers import GPT2Config, GPT2LMHeadModel

from meow_lite import neural, server
from meow_lite.meow import VOCABULARY
from meow_lite.tokenizer import MeowTokenizer

MODEL_DIR = neural.DEFAULT_MODEL_PATH
_VOCAB_SET = set(VOCABULARY)


def _assert_meow_text(text: str) -> None:
    assert text, "generation must be non-empty"
    for raw in text.split():
        word = raw.rstrip(".!?")
        if word:
            assert word.lower() in _VOCAB_SET, f"non-meow word: {raw!r}"


def test_tokenizer_roundtrip():
    tokenizer = MeowTokenizer()
    text = "Meow mrrp purrr."
    ids = tokenizer.encode(text)
    assert ids[0] == tokenizer.bos_token_id
    assert tokenizer.decode(ids, skip_special_tokens=True) == text
    raw_ids = tokenizer.encode(text, add_special_tokens=False)
    assert tokenizer.decode(raw_ids) == text
    assert tokenizer.vocab_size == 2 * len(VOCABULARY) + 3 + 3  # lower+cap, .!?, bos/eos/pad


def test_untrained_forward_logit_shape():
    config = GPT2Config(
        vocab_size=MeowTokenizer().vocab_size,
        n_positions=32,
        n_embd=64,
        n_head=2,
        n_layer=2,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    inputs = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        logits = model(inputs).logits
    assert logits.shape == (1, 8, config.vocab_size)


@pytest.mark.skipif(not __import__("os").path.isdir(MODEL_DIR), reason="checkpoint not trained yet")
def test_trained_checkpoint_generation():
    engine = neural.try_load(MODEL_DIR)
    assert engine is not None, "trained checkpoint must load"
    text = neural.seeded_generate("Explain quantum computing", engine)
    _assert_meow_text(text)


def test_server_with_model_path(monkeypatch):
    monkeypatch.setenv("MEOW_LITE_MODEL_PATH", MODEL_DIR)
    server.reset_engine()
    try:
        client = TestClient(server.app)
        response = client.post(
            "/v1/chat/completions",
            json={"model": "meow-lite", "messages": [{"role": "user", "content": "hello there"}]},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "chat.completion"
        _assert_meow_text(body["choices"][0]["message"]["content"])
        assert server._get_engine() is not None, "server must be using the neural engine"
    finally:
        server.reset_engine()


def test_server_determinism_with_model(monkeypatch):
    monkeypatch.setenv("MEOW_LITE_MODEL_PATH", MODEL_DIR)
    server.reset_engine()
    try:
        client = TestClient(server.app)
        payload = {
            "model": "meow-lite",
            "messages": [{"role": "user", "content": "determinism check"}],
        }
        first = client.post("/v1/chat/completions", json=payload).json()
        second = client.post("/v1/chat/completions", json=payload).json()
        assert (
            first["choices"][0]["message"]["content"]
            == second["choices"][0]["message"]["content"]
        )
        _assert_meow_text(first["choices"][0]["message"]["content"])
    finally:
        server.reset_engine()
