#!/usr/bin/env python
"""v6 training: byte-level BPE tokenizer + from-scratch GPT-2 on teacher data.

Reads data/v6/train.jsonl ({"prompt","response",...}), trains a byte-level
BPE tokenizer (vocab 8000, cat action tokens + [BOS]/[SEP]/[EOS]/[PAD] as
never-split special tokens), then trains a 6-layer/8-head/256-dim GPT-2 with
completion-only loss ([BOS] prompt [SEP] response [EOS], prompt span masked).

Usage:
  /opt/homebrew/bin/uv run python tools/train_v6.py            # full run
  /opt/homebrew/bin/uv run python tools/train_v6.py --smoke    # 32 ex / 5 steps
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from tokenizers.trainers import BpeTrainer
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Config, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meow_lite.tokenizer_v6 import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, SEP_TOKEN  # noqa: E402
from meow_lite.tokenizer import ACTION_TOKENS  # noqa: E402

SPECIAL_TOKENS = list(ACTION_TOKENS) + [BOS_TOKEN, SEP_TOKEN, EOS_TOKEN, PAD_TOKEN]

DEFAULT_VOCAB_SIZE = 8000
DEFAULT_MAX_LEN = 192


def train_tokenizer(texts, vocab_size: int = DEFAULT_VOCAB_SIZE) -> Tokenizer:
    """Byte-level BPE over the corpus; special tokens are added and never split."""
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(texts, trainer)
    return tokenizer


def load_pairs(data_path) -> list[dict]:
    pairs = []
    with open(data_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                if row.get("prompt") and row.get("response"):
                    pairs.append(row)
    return pairs


def encode_pair(wrapper, prompt: str, response: str, max_len: int = DEFAULT_MAX_LEN):
    """[BOS] prompt [SEP] response [EOS] with completion-only labels.

    Returns (input_ids, labels): the prompt span through [SEP] is masked to
    -100; padding is masked too.
    """
    ids = (
        [wrapper.bos_id]
        + wrapper.encode_ids(prompt)
        + [wrapper.sep_id]
        + wrapper.encode_ids(response)
        + [wrapper.eos_id]
    )
    ids = ids[:max_len]
    sep_index = ids.index(wrapper.sep_id)
    labels = list(ids)
    for i in range(sep_index + 1):
        labels[i] = -100
    return ids, labels


def build_tensors(wrapper, pairs, max_len: int = DEFAULT_MAX_LEN):
    """Padded (input_ids, attention_mask, labels) tensors for the whole set."""
    encoded = [encode_pair(wrapper, row["prompt"], row["response"], max_len) for row in pairs]
    longest = max(len(ids) for ids, _ in encoded)
    n = len(encoded)
    input_ids = torch.full((n, longest), wrapper.pad_id, dtype=torch.long)
    attention_mask = torch.zeros((n, longest), dtype=torch.long)
    labels = torch.full((n, longest), -100, dtype=torch.long)
    for i, (ids, row_labels) in enumerate(encoded):
        length = len(ids)
        input_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :length] = 1
        labels[i, :length] = torch.tensor(row_labels, dtype=torch.long)
    return input_ids, attention_mask, labels


def pick_device(preferred: str = "auto") -> str:
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed % (2**63 - 1))


def build_model(vocab_size: int) -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=DEFAULT_MAX_LEN,
        n_embd=256,
        n_head=8,
        n_layer=6,
        bos_token_id=None,
        eos_token_id=None,
    )
    return GPT2LMHeadModel(config)


def run_training(
    data_path,
    out_dir,
    device: str = "auto",
    epochs: int = 2,
    batch_size: int = 64,
    lr: float = 3e-4,
    seed: int = 20261001,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_len: int = DEFAULT_MAX_LEN,
    max_steps: int = 0,
    limit: int = 0,
    log=print,
):
    """Train and save; returns (loss_history, tokenizer, model)."""
    set_seed(seed)
    pairs = load_pairs(data_path)
    if limit:
        pairs = pairs[:limit]
    if not pairs:
        raise RuntimeError(f"no training pairs found in {data_path}")

    texts = [row["prompt"] for row in pairs] + [row["response"] for row in pairs]
    hf_tokenizer = train_tokenizer(texts, vocab_size=vocab_size)
    from meow_lite.tokenizer_v6 import TokenizerV6

    wrapper = TokenizerV6(hf_tokenizer)
    log(f"[tokenizer] vocab_size={wrapper.vocab_size} (requested {vocab_size})")

    input_ids, attention_mask, labels = build_tensors(wrapper, pairs, max_len)
    log(f"[data] {len(pairs)} pairs, tensor shape {tuple(input_ids.shape)}")

    model = build_model(wrapper.vocab_size)
    param_count = sum(p.numel() for p in model.parameters())
    log(f"[model] {param_count} parameters")

    resolved = pick_device(device)
    model.to(resolved)
    dataset = TensorDataset(input_ids, attention_mask, labels)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    steps_per_epoch = max(1, len(loader))
    total_steps = steps_per_epoch * epochs
    if max_steps:
        total_steps = min(total_steps, max_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    losses = []
    model.train()
    step = 0
    done = False
    for epoch in range(epochs):
        for batch_ids, batch_mask, batch_labels in loader:
            if max_steps and step >= max_steps:
                done = True
                break
            batch_ids, batch_mask, batch_labels = (
                batch_ids.to(resolved),
                batch_mask.to(resolved),
                batch_labels.to(resolved),
            )
            optimizer.zero_grad()
            output = model(
                input_ids=batch_ids, attention_mask=batch_mask, labels=batch_labels
            )
            output.loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(output.loss.item())
            step += 1
            if step % 25 == 0:
                log(f"[train] step {step}/{total_steps}: loss {losses[-1]:.4f}")
        if done:
            break

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.save_pretrained(out_path)
    hf_tokenizer.save(str(out_path / "tokenizer.json"))
    log(f"[save] model + tokenizer.json -> {out_path}")
    log(f"[train] {len(losses)} steps, final loss {losses[-1]:.4f}")
    return losses, wrapper, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/v6/train.jsonl")
    parser.add_argument("--out", default="models/meow-lite-v6")
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--smoke", action="store_true", help="32 examples, 5 steps")
    args = parser.parse_args()

    overrides = {}
    if args.smoke:
        overrides = {"limit": 32, "max_steps": 5, "epochs": 2, "batch_size": 8}
    params = {
        "data_path": args.data,
        "out_dir": args.out,
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "vocab_size": args.vocab_size,
        "max_len": args.max_len,
    }
    params.update(overrides)
    run_training(**params)


if __name__ == "__main__":
    main()
