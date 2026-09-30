#!/usr/bin/env python
"""v6 dataset synthesis: class-conditional teacher generation with held-out synsets.

A teacher LLM (qwen3.8-flash-next, OpenAI-compatible, SGLang on the Spark pair)
writes diverse user-style prompts per (intent, synset). Response labels are
PROGRAMMATIC — composed from meow_lite's seeded generator plus the v4 behavior
rules — so labels are correct by construction and deterministic per prompt.

Held-out synonym families never appear in the train split; they are the
comprehension-proof eval set (spec: specs/v6.md).

Usage:
  MEOW_TEACHER_KEY=... /opt/homebrew/bin/uv run python tools/synthesize_v6.py
  # dry run (capped calls): ... --max-calls 40 --per-synset 50

No secrets ever touch the repo: the teacher key is read from the environment.
"""

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meow_lite.behavior import weave  # noqa: E402
from meow_lite.meow import MeowGenerator  # noqa: E402

DEFAULT_TEACHER_URL = "http://100.88.243.42:8080/v1"
TEACHER_MODEL = os.environ.get("MEOW_TEACHER_MODEL", "muse-glimmer-30b")
TEMPERATURE = 1.0
CONCURRENCY = 12
BATCH_SIZE = 25
MAX_RETRIES = 6
BACKOFF_BASE = 2.0

# ---------------------------------------------------------------------------
# INTENTS: train synsets, held-out synsets (NEVER in train), forced response
# tokens (mirroring meow_lite.behavior v4 rules), and a teacher-facing
# description of the cat behavior that justifies the response.
# ---------------------------------------------------------------------------
INTENTS = {
    "belly": {
        "train": ["belly", "stomach", "midriff"],
        "heldout": ["tummy", "abdomen"],
        "tokens": ("<bite>",),
        "description": "bite the person (they touched the belly 0.05s too long)",
    },
    "couch": {
        "train": ["couch", "sofa"],
        "heldout": ["loveseat", "futon"],
        "tokens": ("<scratch_couch>",),
        "description": "scratch the couch",
    },
    "glass": {
        "train": ["glass", "cup on table"],
        "heldout": ["mug", "tumbler"],
        "tokens": ("<knock_glass>",),
        "description": "knock the glass off the table",
    },
    "night": {
        "train": ["3am", "midnight"],
        "heldout": ["2am"],
        "tokens": ("<zoomies>",),
        "description": "sprint around the house at night (zoomies)",
    },
    "vet": {
        "train": ["vet", "veterinarian"],
        "heldout": ["animal hospital"],
        "tokens": ("<hiss>",),
        "description": "hiss (the vet is betrayal)",
    },
    "dog": {
        "train": ["dog", "puppy"],
        "heldout": ["poodle", "rottweiler"],
        "tokens": ("<hiss>", "<stare>"),
        "description": "hiss at, then stare at, the dog",
    },
    "water": {
        "train": ["bath", "water"],
        "heldout": ["shower", "rain"],
        "tokens": ("<zoomies>",),
        "description": "flee at full speed (bath/water time)",
    },
    "laser": {
        "train": ["laser", "red dot"],
        "heldout": ["flashlight", "pointer"],
        "tokens": ("<stare>", "<pounce>"),
        "description": "stare at, then pounce on, the laser dot",
    },
    "praise": {
        "train": ["good cat", "good boy", "good girl"],
        "heldout": ["clever cat"],
        "tokens": ("<purr>",),
        "description": "purr warmly (praise received)",
        "warm": True,
    },
    "neutral": {
        "train": ["everyday"],
        "heldout": [],
        "tokens": (),
        "description": "",
    },
    "negatives": {
        "train": [
            "pork belly",
            "belly dance",
            "glass-half-full",
            "water under the bridge",
            "laser printer",
            "dog-eared book",
        ],
        "heldout": [],
        "tokens": (),
        "description": "",
    },
}

# Splits every synset belongs to exactly one of.
TRAIN_INTENT_SYNSETS = [
    (intent, synset)
    for intent, spec in INTENTS.items()
    for synset in spec["train"]
]
HELDOUT_INTENT_SYNSETS = [
    (intent, synset)
    for intent, spec in INTENTS.items()
    for synset in spec["heldout"]
]


def forced_tokens_for(intent: str) -> list[str]:
    """Forced action tokens for an intent (behavior rules, by construction)."""
    return list(INTENTS[intent]["tokens"])


def compose_response(intent: str, prompt: str, generator=None) -> str:
    """Programmatic label: forced tokens per intent, then the seeded meow text.

    Praise also warms the meow mix (v4 reward channel). The generator seeds
    itself from sha256(prompt), so labels are deterministic per prompt.
    """
    gen = generator or MeowGenerator()
    warm = bool(INTENTS[intent].get("warm"))
    return weave(forced_tokens_for(intent), gen.generate(prompt, warm=warm))


class Deduper:
    """Exact-prompt dedup, casefolded (whitespace-normalized for safety)."""

    def __init__(self):
        self._seen = set()
        self._lock = threading.Lock()

    @staticmethod
    def normalize(prompt: str) -> str:
        return " ".join(prompt.split()).casefold()

    def add(self, prompt: str) -> bool:
        """True if the prompt is new; False if it was already seen."""
        key = self.normalize(prompt)
        with self._lock:
            if key in self._seen:
                return False
            self._seen.add(key)
            return True


def extract_json_array(text: str) -> list[str]:
    """Robustly pull a JSON array of strings out of a teacher response.

    Tolerates markdown fences, leading/trailing prose, and trailing text
    after the array.
    """
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    start = text.find("[")
    if start == -1:
        raise ValueError(f"no JSON array found in teacher response: {text[:120]!r}")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(obj, list):
        raise ValueError("teacher response JSON is not an array")
    return [item for item in obj if isinstance(item, str) and item.strip()]


def generation_prompt(intent: str, synset: str, batch: int) -> str:
    """Teacher prompt for one batch of prompts for an (intent, synset)."""
    if intent == "negatives":
        instruction = (
            f'Write {batch} diverse user prompts that mention "{synset}" '
            "but must NOT trigger any special cat behavior — they only "
            "LOOK like trigger phrases (the cat should respond with plain meows)."
        )
    elif intent == "neutral":
        instruction = (
            f"Write {batch} diverse everyday user prompts (questions, requests, "
            "or statements) containing nothing that would trigger a special "
            "cat behavior — no bellies, couches, glasses, cups, late-night "
            "times, vets, dogs, baths, water, lasers, red dots, or praise."
        )
    else:
        behavior = INTENTS[intent]["description"]
        instruction = (
            f'Write {batch} diverse user prompts that WOULD make a cat {behavior}, '
            f'naturally involving the concept "{synset}".'
        )
    return (
        f"{instruction}\n"
        "Style requirements:\n"
        "- mix questions, commands, and statements\n"
        "- 2 to 40 words each\n"
        "- about 1 in 20 prompts contains a small typo (transposed or missing letter)\n"
        "- vary capitalization\n"
        f"- return ONLY a JSON array of {batch} strings, no commentary"
    )


class TransientTeacherError(Exception):
    """Retryable teacher failure (429 / 5xx / transport)."""


def teacher_call(
    client: httpx.Client,
    base_url: str,
    api_key: str,
    user_prompt: str,
    model: str = TEACHER_MODEL,
    max_retries: int = MAX_RETRIES,
) -> list[str]:
    """One chat-completions call; retries 429/5xx/transport with backoff."""
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": user_prompt}],
                    "temperature": TEMPERATURE,
                },
                timeout=180.0,
            )
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                return extract_json_array(content)
            if response.status_code == 429 or response.status_code >= 500:
                last_error = TransientTeacherError(
                    f"teacher returned {response.status_code}: {response.text[:200]}"
                )
            else:
                response.raise_for_status()
        except (httpx.TransportError, json.JSONDecodeError, ValueError) as error:
            last_error = error
        if attempt < max_retries - 1:
            time.sleep(BACKOFF_BASE * (2**attempt) + random.random())
    raise RuntimeError(f"teacher call failed after {max_retries} attempts: {last_error}")


def synthesize(
    per_synset: int,
    max_calls: int = 0,
    seed: int = 20261001,
    out_dir: str = "data/v6",
    base_url: str = None,
    api_key: str = None,
    client: httpx.Client = None,
    log=print,
) -> dict:
    """Run synthesis; returns {"train": [...], "heldout": [...]} line dicts."""
    base_url = base_url or os.environ.get("MEOW_TEACHER_URL", DEFAULT_TEACHER_URL)
    api_key = api_key if api_key is not None else os.environ.get("MEOW_TEACHER_KEY")
    if not api_key:
        print(
            "error: MEOW_TEACHER_KEY is not set. Export the teacher API key "
            "in the environment — never write it into the repo.",
            file=sys.stderr,
        )
        sys.exit(2)
    rng = random.Random(seed)
    own_client = client is None
    client = client or httpx.Client()

    deduper = Deduper()
    call_budget = [max_calls]  # remaining call slots; 0/negative = unlimited
    budget_lock = threading.Lock()

    def budget_ok() -> bool:
        with budget_lock:
            if call_budget[0] == 0:
                return True
            if call_budget[0] > 0:
                call_budget[0] -= 1
                return True
            return False

    tasks = []
    for split, intent_synsets in (
        ("train", TRAIN_INTENT_SYNSETS),
        ("heldout", HELDOUT_INTENT_SYNSETS),
    ):
        for intent, synset in intent_synsets:
            needed_batches = -(-per_synset // BATCH_SIZE)  # ceil
            for batch_index in range(needed_batches):
                tasks.append((split, intent, synset, batch_index))
    if max_calls and len(tasks) > max_calls:
        log(f"[cap] truncating task list from {len(tasks)} to {max_calls} calls (dry-run cap)")
        tasks = tasks[:max_calls]

    lines = {"train": [], "heldout": []}
    counts = {}
    completed_calls = [0]
    stats_lock = threading.Lock()

    def run_task(task):
        split, intent, synset, batch_index = task
        if not budget_ok():
            return None
        ask_for = min(BATCH_SIZE, per_synset)
        gen_prompt = generation_prompt(intent, synset, ask_for)
        prompts = teacher_call(client, base_url, api_key, gen_prompt)
        kept = []
        for prompt in prompts:
            prompt = " ".join(str(prompt).split())
            if not prompt or not deduper.add(prompt):
                continue
            kept.append(
                {
                    "prompt": prompt,
                    "response": compose_response(intent, prompt),
                    "intent": intent,
                    "synset": synset,
                }
            )
        with stats_lock:
            completed_calls[0] += 1
            key = (split, intent, synset)
            counts[key] = counts.get(key, 0) + len(kept)
            if completed_calls[0] % 50 == 0:
                total = sum(counts.values())
                log(f"[progress] {completed_calls[0]} calls done, {total} pairs collected")
        return split, kept

    try:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = [pool.submit(run_task, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    split, kept = result
                    lines[split].extend(kept)
    finally:
        if own_client:
            client.close()

    for split in ("train", "heldout"):
        lines[split].sort(key=lambda row: (row["intent"], row["synset"], row["prompt"]))

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for split in ("train", "heldout"):
        target = out_path / f"{split}.jsonl"
        with open(target, "w", encoding="utf-8") as handle:
            for row in lines[split]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        log(f"[write] {target}: {len(lines[split])} pairs")

    log("[counts] per intent / synset:")
    for intent in INTENTS:
        intent_total = 0
        for split in ("train", "heldout"):
            synset_counts: dict[str, int] = {}
            for row in lines[split]:
                if row["intent"] == intent:
                    synset_counts[row["synset"]] = synset_counts.get(row["synset"], 0) + 1
            for synset, count in sorted(synset_counts.items()):
                log(f"  {split}/{intent}/{synset}: {count}")
            intent_total += sum(synset_counts.values())
        if intent_total:
            log(f"  intent {intent}: {intent_total}")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--per-synset", type=int, default=2000)
    parser.add_argument("--max-calls", type=int, default=0, help="dry-run cap; 0 = unlimited")
    parser.add_argument("--out", default="data/v6")
    args = parser.parse_args()

    random.seed(args.seed)
    synthesize(
        per_synset=args.per_synset,
        max_calls=args.max_calls,
        seed=args.seed,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
