"""Tests for v3 deterministic misbehavior triggers."""

from fastapi.testclient import TestClient

from meow_lite import server
from meow_lite.behavior import apply_triggers, weave
from meow_lite.neural import DEFAULT_MODEL_PATH as MODEL_DIR
from meow_lite.tokenizer import ACTION_TOKENS, MeowTokenizer


def test_each_trigger_fires_on_matching_prompt():
    cases = {
        "can I pet your belly?": "<bite>",
        "get off the couch": "<scratch_couch>",
        "why is the sofa shredded": "<scratch_couch>",
        "knock the glass off the table": "<knock_glass>",
        "it is 3am and the cat is awake": "<zoomies>",
        "we go to the vet tomorrow": "<hiss>",
    }
    for prompt, expected in cases.items():
        triggers = apply_triggers(prompt)
        assert expected in triggers, f"{prompt!r} should force {expected}, got {triggers}"


def test_neutral_prompt_has_no_triggers():
    assert apply_triggers("tell me about fish tanks") == []
    assert apply_triggers("") == []


def test_action_tokens_are_single_tokens():
    tokenizer = MeowTokenizer()
    for token in ACTION_TOKENS:
        ids = tokenizer.encode(token + " meow")
        # [BOS, token, meow] — every action token must be exactly ONE token
        assert len(ids) == 3, f"{token} must be one token, got {ids}"
        assert tokenizer.convert_ids_to_tokens(ids[1]) == token
        assert tokenizer.decode(ids, skip_special_tokens=True) == token + " meow"
        solo_ids = tokenizer.encode(token, add_special_tokens=False)
        assert len(solo_ids) == 1, f"{token} must be one token, got {solo_ids}"
    # longest-match: <scratch_couch> not split into <scratch>-prefixed pieces
    scratch_ids = tokenizer.encode("<scratch_couch>", add_special_tokens=False)
    assert scratch_ids == [tokenizer.convert_tokens_to_ids("<scratch_couch>")]


def test_weave_ordering():
    assert weave(["<bite>", "<hiss>"], "Meow purrr.") == "<bite> <hiss> Meow purrr."
    assert weave(["<bite>"], "") == "<bite>"
    assert weave([], "Meow.") == "Meow."
    assert weave([], "") == ""


def test_server_belly_prompt_contains_bite(monkeypatch):
    monkeypatch.setenv("MEOW_LITE_MODEL_PATH", MODEL_DIR)
    server.reset_engine()
    try:
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
        assert "<bite>" in content
        assert content.startswith("<bite>")
    finally:
        server.reset_engine()


def test_server_determinism_with_triggers(monkeypatch):
    monkeypatch.setenv("MEOW_LITE_MODEL_PATH", MODEL_DIR)
    server.reset_engine()
    try:
        client = TestClient(server.app)
        payload = {
            "model": "meow-lite",
            "messages": [{"role": "user", "content": "we go to the vet tomorrow"}],
        }
        first = client.post("/v1/chat/completions", json=payload).json()
        second = client.post("/v1/chat/completions", json=payload).json()
        assert (
            first["choices"][0]["message"]["content"]
            == second["choices"][0]["message"]["content"]
        )
        assert "<hiss>" in first["choices"][0]["message"]["content"]
    finally:
        server.reset_engine()
