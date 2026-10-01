"""Tests for the v6 synthesis pipeline (all HTTP mocked)."""

import httpx
import pytest

from meow_lite.tokenizer import ACTION_TOKENS
from tools import synthesize_v6 as syn


def test_extract_json_array_robustness():
    assert syn.extract_json_array('["a", "b"]') == ["a", "b"]
    assert syn.extract_json_array('```json\n["a", "b"]\n```') == ["a", "b"]
    assert syn.extract_json_array('```\n["a"]\n```') == ["a"]
    assert syn.extract_json_array('Sure! Here you go:\n["a", "b"]') == ["a", "b"]
    assert syn.extract_json_array('["a", "b"]\n\nLet me know if you need more.') == ["a", "b"]
    assert syn.extract_json_array('blah ["a \\"quoted\\"", "b"] trailing') == ["a \"quoted\"", "b"]
    with pytest.raises(ValueError):
        syn.extract_json_array("no array here")
    with pytest.raises(ValueError):
        syn.extract_json_array('{"not": "an array"}')


def test_split_assignment_invariant():
    train_synsets = {s for _, s in syn.TRAIN_INTENT_SYNSETS}
    heldout_synsets = {s for _, s in syn.HELDOUT_INTENT_SYNSETS}
    assert not (train_synsets & heldout_synsets), "held-out synsets must never appear in train"
    # each intent's own lists must be disjoint too
    for intent, spec in syn.INTENTS.items():
        assert not (set(spec["train"]) & set(spec["heldout"])), intent
    # spot-check the spec table
    assert "tummy" in heldout_synsets and "tummy" not in train_synsets
    assert "belly" in train_synsets
    assert "2am" in heldout_synsets
    assert "poodle" in heldout_synsets and "dog" in train_synsets


def test_response_composition_matches_behavior_rules():
    gen = syn.MeowGenerator()
    expected_tokens = {
        "belly": ["<bite>"],
        "couch": ["<scratch_couch>"],
        "glass": ["<knock_glass>"],
        "night": ["<zoomies>"],
        "vet": ["<hiss>"],
        "dog": ["<hiss>", "<stare>"],
        "water": ["<zoomies>"],
        "laser": ["<stare>", "<pounce>"],
    }
    for intent, tokens in expected_tokens.items():
        response = syn.compose_response(intent, "a prompt for this intent", generator=gen)
        assert response.startswith(" ".join(tokens)), f"{intent}: {response!r}"
        assert response[len(" ".join(tokens)) :].strip()  # meow text follows
    # praise leads with <purr>
    praise = syn.compose_response("praise", "who's a good cat", generator=gen)
    assert praise.startswith("<purr>")
    # determinism: same prompt -> same response
    assert syn.compose_response("belly", "fixed prompt", generator=gen) == syn.compose_response(
        "belly", "fixed prompt", generator=gen
    )


def test_negatives_and_neutral_get_no_forced_tokens():
    assert syn.forced_tokens_for("negatives") == []
    assert syn.forced_tokens_for("neutral") == []
    for intent in ("negatives", "neutral"):
        response = syn.compose_response(intent, "pork belly recipe", generator=syn.MeowGenerator())
        for token in ACTION_TOKENS:
            assert token not in response, f"{intent} leaked {token}: {response!r}"


def test_dedup_casefold():
    deduper = syn.Deduper()
    assert deduper.add("Can I rub the belly?") is True
    assert deduper.add("can i rub the belly?") is False
    assert deduper.add("  Can I rub the belly?  ") is False
    assert deduper.add("totally different") is True


def test_teacher_call_retries_on_429_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="slow down")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '["one prompt", "two prompts"]'}}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    prompts = syn.teacher_call(client, "http://teacher.test/v1", "test-key", "give prompts")
    assert prompts == ["one prompt", "two prompts"]
    assert calls["n"] == 2


def test_teacher_call_retries_on_5xx_and_gives_up():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="overloaded")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError):
        syn.teacher_call(
            client, "http://teacher.test/v1", "test-key", "give prompts", max_retries=2
        )


def test_teacher_call_fails_loudly_on_4xx():
    client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(401)))
    with pytest.raises(httpx.HTTPStatusError):
        syn.teacher_call(client, "http://teacher.test/v1", "bad-key", "give prompts")


def test_missing_api_key_fails(monkeypatch, capsys):
    monkeypatch.delenv("MEOW_TEACHER_KEY", raising=False)
    monkeypatch.setattr(syn, "httpx", syn.httpx)  # no-op, keep surface explicit
    with pytest.raises(SystemExit):
        syn.synthesize(per_synset=1, client=httpx.Client())
    assert "MEOW_TEACHER_KEY" in capsys.readouterr().err


def test_generation_prompt_styles():
    belly = syn.generation_prompt("belly", "tummy", 25)
    assert '"tummy"' in belly and "JSON array of 25" in belly
    negative = syn.generation_prompt("negatives", "pork belly", 25)
    assert "NOT trigger" in negative and '"pork belly"' in negative
    neutral = syn.generation_prompt("neutral", "everyday", 25)
    assert "no bellies" in neutral


def test_synthesize_end_to_end_mocked(tmp_path, monkeypatch):
    """Small mocked run: real HTTP mocked, files written, split invariant holds."""
    seen_batches = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_batches["n"] += 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '["prompt %d unique one", "prompt %d unique two", "prompt %d unique three"]'
                            % (seen_batches["n"], seen_batches["n"], seen_batches["n"])
                        }
                    }
                ]
            },
        )

    monkeypatch.setenv("MEOW_TEACHER_KEY", "test-key")
    lines = syn.synthesize(
        per_synset=3,
        out_dir=str(tmp_path),
        base_url="http://teacher.test/v1",
        api_key="test-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        log=lambda *_: None,
    )
    all_rows = lines["train"] + lines["heldout"]
    assert all_rows, "mocked run must produce pairs"
    train_keys = {(row["intent"], row["synset"]) for row in lines["train"]}
    heldout_keys = {(row["intent"], row["synset"]) for row in lines["heldout"]}
    assert not (train_keys & heldout_keys)
    for row in all_rows:
        assert set(row) == {"prompt", "response", "intent", "synset"}
    # files exist
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "heldout.jsonl").exists()
