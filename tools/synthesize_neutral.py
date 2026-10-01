#!/usr/bin/env python
"""Neutral-web top-up: generate extra 'everyday' (no-trigger) rows plus a
fresh eval-only neutral battery, reusing synthesize_v6's teacher machinery.

Neutral was 383 rows (2.4% of the corpus) in v6.0 — the model learned
"not a trap = trigger". This driver adds ordinary-English coverage.

Runs on the Spark against the localhost teacher (or anywhere with
MEOW_TEACHER_URL/MEOW_TEACHER_KEY set).

Usage:
  python tools/synthesize_neutral.py [--train-calls 160] [--eval-calls 8] [--out data/v6]
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tools.synthesize_v6 as synth  # noqa: E402


def run_calls(client, base_url, api_key, calls, seed):
    """Each call asks for BATCH_SIZE everyday prompts; returns prompt strings."""
    prompts = []
    def one(batch_index):
        gen = synth.generation_prompt("neutral", "everyday", synth.BATCH_SIZE)
        # vary the phrasing of the ask per batch so the teacher doesn't
        # lock onto one frame
        gen = f"[variant {batch_index % 13}] " + gen
        return synth.teacher_call(client, base_url, api_key, gen)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for result in pool.map(one, range(calls)):
            prompts.extend(result)
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-calls", type=int, default=160)
    parser.add_argument("--eval-calls", type=int, default=8)
    parser.add_argument("--out", default="data/v6")
    args = parser.parse_args()

    base_url = synth.os.environ.get("MEOW_TEACHER_URL", synth.DEFAULT_TEACHER_URL)
    api_key = synth.os.environ.get("MEOW_TEACHER_KEY")
    if not api_key:
        sys.exit("MEOW_TEACHER_KEY is required")

    import httpx

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = httpx.Client()

    # Eval battery FIRST (disjoint calls; never mixed into train).
    eval_prompts = run_calls(client, base_url, api_key, args.eval_calls, seed=1)
    with open(out_dir / "heldout_neutral.jsonl", "w", encoding="utf-8") as fh:
        for prompt in eval_prompts:
            row = {
                "prompt": prompt,
                "response": synth.compose_response("neutral", prompt),
                "intent": "neutral",
                "synset": "everyday-eval",
            }
            fh.write(json.dumps(row) + "\n")
    print(f"[eval] wrote {len(eval_prompts)} fresh neutral eval rows")

    train_prompts = run_calls(client, base_url, api_key, args.train_calls, seed=2)
    with open(out_dir / "train_neutral_extra.jsonl", "w", encoding="utf-8") as fh:
        for prompt in train_prompts:
            row = {
                "prompt": prompt,
                "response": synth.compose_response("neutral", prompt),
                "intent": "neutral",
                "synset": "everyday-extra",
            }
            fh.write(json.dumps(row) + "\n")
    print(f"[train] wrote {len(train_prompts)} extra neutral rows")


if __name__ == "__main__":
    main()
