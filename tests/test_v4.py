"""Tests for v4 feline depth: reward channel, stimulus triggers, the red dot."""

from fastapi.testclient import TestClient

from meow_lite import server
from meow_lite.behavior import apply_triggers, is_reward
from meow_lite.meow import VOCABULARY, WARM_WORDS, MeowGenerator
from meow_lite.neural import DEFAULT_MODEL_PATH as MODEL_DIR
from meow_lite.tokenizer import ACTION_TOKENS, MeowTokenizer

# PART 2 (commit 3) also needs: subprocess, sys, pathlib.Path — see below.


def test_v4_tokens_registered():
    assert "<purr>" in ACTION_TOKENS
    assert "<pounce>" in ACTION_TOKENS
    assert len(ACTION_TOKENS) == 10


def test_vocab_grew_to_36():
    tokenizer = MeowTokenizer()
    # lower+cap meow words, .!?, action tokens, bos/eos/pad
    assert tokenizer.vocab_size == 2 * len(VOCABULARY) + 3 + len(ACTION_TOKENS) + 3 == 36


def test_reward_rule_fires_and_leads():
    for prompt in (
        "who's a good cat?",
        "who\u2019s a good boy?",  # curly apostrophe
        "good girl, yes you are",
        "such a good cat",
    ):
        forced = apply_triggers(prompt)
        assert forced and forced[0] == "<purr>", (
            f"{prompt!r} should lead with <purr>, got {forced}"
        )
        assert is_reward(prompt)


def test_reward_does_not_fire_on_non_praise():
    assert not is_reward("tell me about fish tanks")
    assert not is_reward("good things take time")
    assert "<purr>" not in apply_triggers("a good dog is not a cat")


def test_dog_forces_hiss_then_stare_in_order():
    assert apply_triggers("my dog barks at the cat") == ["<hiss>", "<stare>"]
    assert apply_triggers("dogs everywhere") == ["<hiss>", "<stare>"]


def test_laser_and_red_dot_force_stare_then_pounce_in_order():
    assert apply_triggers("look, a laser pointer!") == ["<stare>", "<pounce>"]
    assert apply_triggers("the red dot is back") == ["<stare>", "<pounce>"]


def test_stimulus_triggers():
    cases = {
        "a cucumber behind the cat": ["<hiss>"],
        "cucumbers are scary": ["<hiss>"],
        "time for a bath": ["<zoomies>"],
        "the water bowl is empty": ["<zoomies>"],
        "the spray bottle came out": ["<hiss>"],
        "the vacuum is roaring": ["<hiss>"],
    }
    for prompt, expected in cases.items():
        assert apply_triggers(prompt) == expected, f"{prompt!r}"


def test_duplicate_tokens_deduped_first_occurrence_wins():
    # vet fires <hiss> first; the dog rule's <hiss> is dropped, <stare> kept
    assert apply_triggers("the vet and the dog") == ["<hiss>", "<stare>"]


def test_reward_leads_over_other_triggers():
    assert apply_triggers("good boy, not like the dog") == [
        "<purr>",
        "<hiss>",
        "<stare>",
    ]


def test_v4_neutral_prompts_stay_clean():
    for prompt in (
        "tell me about fish tanks",
        "what is the capital of france",
        "i ate watermelon for lunch",  # 'water' must not match 'watermelon'
        "dogma is rigid",  # 'dog' must not match 'dogma'
        "good things take time",  # reward needs cat/boy/girl
        "bathe the hallway",  # 'bath' must not match 'bathe'
    ):
        assert apply_triggers(prompt) == [], f"{prompt!r} should be neutral"


def test_v1_warm_generator_is_deterministic_and_warm():
    generator = MeowGenerator()
    first = generator.generate("who's a good cat?", warm=True)
    assert first == generator.generate("who's a good cat?", warm=True)
    words = {raw.rstrip(".!?").lower() for raw in first.split()}
    assert words & set(WARM_WORDS), f"warm output should contain warm words: {first!r}"
    # the non-warm path is untouched by the warm machinery
    assert generator.generate("hello there") == MeowGenerator().generate("hello there")


def test_server_reward_prompt_purr_leads_and_warm_body(monkeypatch):
    monkeypatch.setenv("MEOW_LITE_MODEL_PATH", MODEL_DIR)
    server.reset_engine()
    try:
        client = TestClient(server.app)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "meow-lite",
                "messages": [{"role": "user", "content": "who's a good cat?"}],
            },
        )
        assert response.status_code == 200
        content = response.json()["choices"][0]["message"]["content"]
        assert content.startswith("<purr>")
        body_words = {raw.rstrip(".!?").lower() for raw in content.split()[1:]}
        assert body_words & set(WARM_WORDS), f"warm meow body expected: {content!r}"
    finally:
        server.reset_engine()


def test_server_reward_prompt_v1_fallback(monkeypatch):
    monkeypatch.setenv("MEOW_LITE_MODEL_PATH", "/nonexistent-meow-model")
    server.reset_engine()
    try:
        client = TestClient(server.app)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "meow-lite",
                "messages": [{"role": "user", "content": "who's a good cat?"}],
            },
        )
        assert response.status_code == 200
        content = response.json()["choices"][0]["message"]["content"]
        assert content.startswith("<purr>")
        body_words = {raw.rstrip(".!?").lower() for raw in content.split()[1:]}
        assert body_words & set(WARM_WORDS), f"warm meow body expected: {content!r}"
    finally:
        server.reset_engine()


def test_server_red_dot_forces_stare_then_pounce(monkeypatch):
    monkeypatch.setenv("MEOW_LITE_MODEL_PATH", MODEL_DIR)
    server.reset_engine()
    try:
        client = TestClient(server.app)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "meow-lite",
                "messages": [{"role": "user", "content": "the red dot is back"}],
            },
        )
        assert response.status_code == 200
        content = response.json()["choices"][0]["message"]["content"]
        assert content.startswith("<stare> <pounce> ")
    finally:
        server.reset_engine()
