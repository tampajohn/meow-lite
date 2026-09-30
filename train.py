#!/usr/bin/env python
"""Train the tiny meow GPT-2 from scratch and save it to models/meow-lite/.

Deterministic and CPU-friendly (~1-2 minutes): fixed-seed corpus of 20k meow
sentences, 8k misbehavior sentences (action tokens mixed with meow words) and
4k v4 reward sentences (a <purr> at a random position over warm-weighted
meow words),
word-level vocab (36 tokens), 2-layer/2-head/64-dim GPT-2.

Usage: /opt/homebrew/bin/uv run python train.py
"""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Config, GPT2LMHeadModel

from meow_lite.meow import TERMINALS, VOCABULARY, warm_weights
from meow_lite.neural import DEFAULT_MODEL_PATH
from meow_lite.tokenizer import ACTION_TOKENS, MeowTokenizer

SEED = 20260930
CORPUS_SIZE = 20000
MISBEHAVIOR_SIZE = 8000
REWARD_SIZE = 4000
BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 1e-3
N_LAYER = 2
N_HEAD = 2
N_EMBD = 64
N_POSITIONS = 32
OUTPUT_DIR = DEFAULT_MODEL_PATH


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed % (2**63 - 1))


def make_corpus(count: int, rng: random.Random) -> list[str]:
    sentences = []
    for _ in range(count):
        word_count = rng.randint(3, 12)
        words = [rng.choice(VOCABULARY) for _ in range(word_count)]
        sentence = " ".join(words)
        sentence = sentence[0].upper() + sentence[1:] + rng.choice(TERMINALS)
        sentences.append(sentence)
    return sentences


def make_misbehavior(count: int, rng: random.Random) -> list[str]:
    """Meow sentences with 1-3 action tokens mixed in, e.g.
    '<zoomies> mrrp mraow <knock_glass> !' or 'Mew <hiss> mrrrow .'."""
    sentences = []
    for _ in range(count):
        word_count = rng.randint(3, 12)
        words = [rng.choice(VOCABULARY) for _ in range(word_count)]
        for _ in range(rng.randint(1, 3)):
            words.insert(rng.randint(0, len(words)), rng.choice(ACTION_TOKENS))
        sentence = " ".join(words)
        sentence = sentence[0].upper() + sentence[1:] + rng.choice(TERMINALS)
        sentences.append(sentence)
    return sentences


def make_reward(count: int, rng: random.Random) -> list[str]:
    """Praise-response sentences: a <purr> inserted at a random position over
    warm-weighted meow words, e.g. 'purrr <purr> mrrp prrrt .'. Teaches the
    model the v4 reward channel: <purr> in context with the warm vocabulary."""
    sentences = []
    for _ in range(count):
        word_count = rng.randint(3, 12)
        words = rng.choices(VOCABULARY, weights=warm_weights(), k=word_count)
        words.insert(rng.randint(0, len(words)), "<purr>")
        sentence = " ".join(words) + rng.choice(TERMINALS)
        sentences.append(sentence)
    return sentences


def main() -> None:
    set_seeds(SEED)
    corpus_rng = random.Random(SEED)
    misbehavior_rng = random.Random(SEED + 1)
    reward_rng = random.Random(SEED + 2)
    corpus = (
        make_corpus(CORPUS_SIZE, corpus_rng)
        + make_misbehavior(MISBEHAVIOR_SIZE, misbehavior_rng)
        + make_reward(REWARD_SIZE, reward_rng)
    )
    print(f"corpus: {len(corpus)} sentences, e.g. {corpus[0]!r}")
    print(f"misbehavior example: {corpus[CORPUS_SIZE]!r}")
    print(f"reward example: {corpus[-1]!r}")

    tokenizer = MeowTokenizer()
    print(f"vocab: {tokenizer.vocab_size} tokens")

    encoded = [
        tokenizer.encode(sentence) + [tokenizer.eos_token_id] for sentence in corpus
    ]
    max_len = min(max(len(ids) for ids in encoded), N_POSITIONS)
    pad_id = tokenizer.pad_token_id

    input_ids = torch.full((len(encoded), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long)
    labels = torch.full((len(encoded), max_len), -100, dtype=torch.long)
    for row, ids in enumerate(encoded):
        trimmed = ids[:max_len]
        length = len(trimmed)
        input_ids[row, :length] = torch.tensor(trimmed, dtype=torch.long)
        attention_mask[row, :length] = 1
        labels[row, :length] = input_ids[row, :length]

    dataset = TensorDataset(input_ids, attention_mask, labels)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)

    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=N_POSITIONS,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"model: {param_count} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()
    step = 0
    for epoch in range(EPOCHS):
        running = 0.0
        for batch_input_ids, batch_mask, batch_labels in loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_mask,
                labels=batch_labels,
            )
            outputs.loss.backward()
            optimizer.step()
            running += outputs.loss.item()
            step += 1
            if step % 100 == 0:
                print(f"epoch {epoch + 1} step {step}: loss {running / 100:.4f}")
                running = 0.0

    model.eval()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"saved checkpoint to {OUTPUT_DIR}")

    engine = {"model": model, "tokenizer": tokenizer}
    from meow_lite.behavior import apply_triggers, is_reward, weave
    from meow_lite.neural import seeded_generate

    for prompt in (
        "Explain gravity",
        "can I pet your belly?",
        "it is 3am, what now?",
        "who's a good cat?",
        "look, a red dot!",
    ):
        warm = is_reward(prompt)
        print(
            f"sample {prompt!r} -> "
            f"{weave(apply_triggers(prompt), seeded_generate(prompt, engine, warm=warm))!r}"
        )


if __name__ == "__main__":
    main()
