"""Tests for the v6 training + eval stack (no teacher, no real dataset)."""

import json

import pytest
import torch

from meow_lite import neural_v6, tokenizer_v6
from meow_lite.tokenizer import ACTION_TOKENS
from meow_lite.tokenizer_v6 import BOS_TOKEN, EOS_TOKEN, SEP_TOKEN
from tools import synthesize_v6 as syn
from tools import train_v6 as tv6

SMOKE_STEPS = 5

# 32 deterministic examples via the programmatic labeler (no teacher calls).
FIXTURE_PLAN = [
    ("belly", "belly"), ("belly", "stomach"), ("couch", "couch"), ("couch", "sofa"),
    ("glass", "glass"), ("glass", "cup on table"), ("night", "3am"), ("night", "midnight"),
    ("vet", "vet"), ("vet", "veterinarian"), ("dog", "dog"), ("dog", "puppy"),
    ("water", "bath"), ("water", "water"), ("laser", "laser"), ("laser", "red dot"),
    ("praise", "good cat"), ("praise", "good boy"), ("neutral", "everyday"),
    ("negatives", "pork belly"), ("negatives", "glass-half-full"),
    ("negatives", "water under the bridge"), ("negatives", "laser printer"),
    ("negatives", "dog-eared book"), ("belly", "midriff"), ("couch", "sofa"),
    ("night", "3am"), ("vet", "vet"), ("dog", "puppy"), ("water", "bath"),
    ("praise", "good girl"), ("neutral", "everyday"),
]


def _fixture_rows():
    rows = []
    for i, (intent, synset) in enumerate(FIXTURE_PLAN):
        prompt = f"example {i}: {synset} question number {i}?"
        rows.append(
            {
                "prompt": prompt,
                "response": syn.compose_response(intent, prompt),
                "intent": intent,
                "synset": synset,
            }
        )
    return rows


@pytest.fixture(scope="module")
def smoke_checkpoint(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("v6-smoke") / "meow-lite-v6"
    data_path = tmp_path_factory.mktemp("v6-data") / "train.jsonl"
    with open(data_path, "w", encoding="utf-8") as handle:
        for row in _fixture_rows():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    losses, wrapper, _model = tv6.run_training(
        data_path=data_path,
        out_dir=out_dir,
        device="cpu",
        epochs=2,
        batch_size=8,
        seed=20261001,
        max_steps=SMOKE_STEPS,
        log=lambda *_: None,
    )
    return out_dir, losses


def test_tokenizer_roundtrip_and_action_tokens_single_id(smoke_checkpoint):
    out_dir, _ = smoke_checkpoint
    tokenizer = tokenizer_v6.load(out_dir)
    text = "example prompt about the sofa [SEP] <purr> Meow purrr."
    ids = tokenizer.encode_ids(text)
    assert tokenizer.decode(ids) == text
    for token in ACTION_TOKENS:
        token_ids = tokenizer.encode_ids(token)
        assert token_ids == [tokenizer.token_to_id(token)], f"{token} split into {token_ids}"
        assert tokenizer.decode(token_ids) == token


def test_label_masking_completion_only(smoke_checkpoint):
    out_dir, _ = smoke_checkpoint
    tokenizer = tokenizer_v6.load(out_dir)
    input_ids, labels = tv6.encode_pair(tokenizer, "pet the cat now", "<purr> Meow purrr.")
    sep_index = input_ids.index(tokenizer.sep_id)
    assert input_ids[0] == tokenizer.bos_id
    assert input_ids[sep_index] == tokenizer.sep_id
    assert input_ids[-1] == tokenizer.eos_id
    # prompt span through [SEP] fully masked
    assert all(label == -100 for label in labels[: sep_index + 1])
    # response span carries real labels
    assert all(label != -100 for label in labels[sep_index + 1 :])


def test_smoke_training_reduces_loss(smoke_checkpoint):
    _, losses = smoke_checkpoint
    assert len(losses) == SMOKE_STEPS
    assert losses[-1] < losses[0], f"loss did not drop: {losses}"


def test_cat_mask_leaves_only_cat_tokens_legal(smoke_checkpoint):
    out_dir, _ = smoke_checkpoint
    engine = neural_v6.load(out_dir)
    assert engine is not None
    tokenizer = engine["tokenizer"]
    processor = engine["cat_mask"]
    vocab_size = engine["model"].config.vocab_size
    scores = torch.randn(1, vocab_size)
    input_ids = torch.tensor([[tokenizer.bos_id, 5, tokenizer.sep_id]])
    masked = processor(input_ids, scores)[0]
    allowed = set(neural_v6.cat_token_ids(tokenizer))
    for index in range(vocab_size):
        if index in allowed:
            assert torch.isfinite(masked[index]), f"cat token {index} was masked"
        else:
            assert masked[index] == float("-inf"), f"non-cat token {index} stayed legal"
    # [EOS] must be reachable
    assert tokenizer.eos_id in allowed


def test_seeded_generate_deterministic_with_smoke_checkpoint(smoke_checkpoint):
    out_dir, _ = smoke_checkpoint
    engine = neural_v6.load(out_dir)
    prompt = "can I rub her tummy?"
    first = neural_v6.seeded_generate(prompt, engine)
    second = neural_v6.seeded_generate(prompt, engine)
    assert first == second
    assert first, "generation must be non-empty"
    # decode strips the structural specials
    assert BOS_TOKEN not in first and EOS_TOKEN not in first and SEP_TOKEN not in first
