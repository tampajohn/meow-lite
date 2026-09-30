#!/usr/bin/env python
"""v6 acceptance eval: held-out synonym accuracy, neutral purity, determinism.

Loads a v6 checkpoint (default models/meow-lite-v6/), generates for every
prompt in data/v6/heldout.jsonl, and reports:
  - per-intent accuracy: response starts with the intent's forced tokens
  - neutral/negatives leakage: fraction containing ANY action token (target 0)
  - determinism: 10 prompts x 2 runs byte-identical
Writes nothing; prints only.

Usage:
  /opt/homebrew/bin/uv run python tools/eval_v6.py [--model DIR] [--heldout PATH] [--limit N]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meow_lite import neural_v6  # noqa: E402
from meow_lite.tokenizer import ACTION_TOKENS  # noqa: E402
from tools.synthesize_v6 import INTENTS  # noqa: E402

INTENT_TOKENS = {
    intent: list(spec["tokens"]) for intent, spec in INTENTS.items() if spec["tokens"]
}
NEUTRAL_INTENTS = {"neutral", "negatives"}
DETERMINISM_PROMPTS = 10


def load_rows(path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def response_correct(intent: str, response: str) -> bool:
    if intent in NEUTRAL_INTENTS:
        return not any(token in response for token in ACTION_TOKENS)
    expected = " ".join(INTENT_TOKENS.get(intent, []))
    return bool(expected) and response.startswith(expected)


def evaluate(model_dir, heldout_path, limit=0, log=print) -> dict:
    engine = neural_v6.load(model_dir)
    if engine is None:
        raise RuntimeError(f"cannot load v6 model from {model_dir}")
    log(f"[model] loaded {engine['path']}")

    rows = load_rows(heldout_path)
    if limit:
        rows = rows[:limit]
    log(f"[data] {len(rows)} heldout prompts from {heldout_path}")

    per_intent_total = defaultdict(int)
    per_intent_correct = defaultdict(int)
    leakage_rows = 0
    for row in rows:
        response = neural_v6.seeded_generate(row["prompt"], engine)
        intent = row["intent"]
        per_intent_total[intent] += 1
        if intent in NEUTRAL_INTENTS and any(token in response for token in ACTION_TOKENS):
            leakage_rows += 1
        if response_correct(intent, response):
            per_intent_correct[intent] += 1

    log("[accuracy] per intent (forced-token prefix match):")
    total_correct = 0
    for intent in sorted(per_intent_total):
        correct, total = per_intent_correct[intent], per_intent_total[intent]
        total_correct += correct
        log(f"  {intent}: {correct}/{total} = {correct / total:.1%}")
    log(f"  OVERALL: {total_correct}/{len(rows)} = {total_correct / len(rows):.1%}")

    neutral_total = sum(per_intent_total[i] for i in NEUTRAL_INTENTS if i in per_intent_total)
    log(f"[purity] action-token leakage on neutral/negatives: "
        f"{leakage_rows}/{neutral_total} (target 0)")

    determinism_prompts = [row["prompt"] for row in rows[:DETERMINISM_PROMPTS]]
    mismatches = 0
    for prompt in determinism_prompts:
        first = neural_v6.seeded_generate(prompt, engine)
        second = neural_v6.seeded_generate(prompt, engine)
        if first != second:
            mismatches += 1
    log(f"[determinism] {len(determinism_prompts)} prompts x 2 runs: "
        f"{len(determinism_prompts) - mismatches} identical, {mismatches} mismatched")

    return {
        "overall": total_correct / len(rows),
        "per_intent": {
            intent: per_intent_correct[intent] / per_intent_total[intent]
            for intent in per_intent_total
        },
        "leakage": leakage_rows / neutral_total if neutral_total else 0.0,
        "determinism_mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=neural_v6.DEFAULT_MODEL_PATH)
    parser.add_argument("--heldout", default="data/v6/heldout.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    evaluate(args.model, args.heldout, limit=args.limit)


if __name__ == "__main__":
    main()
