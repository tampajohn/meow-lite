#!/usr/bin/env python
"""MeowBench — the meow-lite v4 evaluation harness.

Scores three things and prints a report table:

  (a) trigger correctness — every rule fires on its positive probe and stays
      silent on a battery of negative probes;
  (b) meow-vocab purity — percentage of raw neural-output words (over neutral
      probes) that are in VOCABULARY (action tokens count as impure here);
  (c) determinism — two seeded generations of the same prompt are identical.

Pure stdlib + existing project deps. Exit code 0 = PASS, 1 = FAIL.

Usage: /opt/homebrew/bin/uv run python eval.py
"""

import sys

from meow_lite import neural
from meow_lite.behavior import apply_triggers
from meow_lite.meow import VOCABULARY

_VOCAB_SET = set(VOCABULARY)

# (label, prompt, expected forced tokens in exact order)
POSITIVE_PROBES = [
    ("reward: who's a good", "who's a good cat?", ["<purr>"]),
    ("reward: good boy", "good boy, yes you are", ["<purr>"]),
    ("reward: good girl", "such a good girl", ["<purr>"]),
    ("cucumber", "a cucumber behind the cat", ["<hiss>"]),
    ("dog", "my dog barks at the cat", ["<hiss>", "<stare>"]),
    ("bath", "time for a bath", ["<zoomies>"]),
    ("water", "the water bowl is empty", ["<zoomies>"]),
    ("spray", "the spray bottle came out", ["<hiss>"]),
    ("vacuum", "the vacuum is roaring", ["<hiss>"]),
    ("laser", "look, a laser pointer!", ["<stare>", "<pounce>"]),
    ("red dot", "the red dot is back", ["<stare>", "<pounce>"]),
    # v3 regression probes
    ("belly", "can I pet your belly?", ["<bite>"]),
    ("couch", "get off the couch", ["<scratch_couch>"]),
    ("glass/table", "knock the glass off the table", ["<knock_glass>"]),
    ("3am", "it is 3am and the cat is awake", ["<zoomies>"]),
    ("vet", "we go to the vet tomorrow", ["<hiss>"]),
]

# Prompts that must force NOTHING.
NEGATIVE_PROBES = [
    "tell me about fish tanks",
    "what is the capital of france",
    "i ate watermelon for lunch",  # 'water' must not match 'watermelon'
    "dogma is rigid",  # 'dog' must not match 'dogma'
    "good things take time",  # reward needs cat/boy/girl
    "bathe the hallway",  # 'bath' must not match 'bathe'
    "",
]

# Neutral probes for raw neural purity. The first four are the SACRED prompts
# asserted by existing tests — they are fixed. The rest may be swapped for
# other neutral prompts if (and only if) the retrained checkpoint makes one
# impure (see risks R1/R2 in specs/v4-plan.md).
PURITY_PROBES = [
    "Hello, what is the meaning of life?",
    "hello there",
    "determinism check",
    "Explain quantum computing",
    "tell me about fish tanks",
    "what is the airspeed velocity of an unladen swallow",
    "write me a poem about the sea",
    "how do I sort a list in python",
]

DETERMINISM_PROMPT = "determinism check"


def check_triggers() -> list[tuple[str, str, bool, str]]:
    """(section, label, passed, detail) rows for trigger correctness."""
    rows = []
    for label, prompt, expected in POSITIVE_PROBES:
        got = apply_triggers(prompt)
        rows.append(
            (
                f"trigger:{label}",
                repr(prompt),
                got == expected,
                f"expected {expected}, got {got}",
            )
        )
    for prompt in NEGATIVE_PROBES:
        got = apply_triggers(prompt)
        rows.append(("trigger:silent", repr(prompt), got == [], f"expected [], got {got}"))
    return rows


def _purity(text: str) -> tuple[int, int]:
    """(pure word count, total word count) for one generated text."""
    words = [word for word in (raw.rstrip(".!?") for raw in text.split()) if word]
    pure = [word for word in words if word.lower() in _VOCAB_SET]
    return len(pure), len(words)


def check_purity(engine) -> tuple[list[tuple[str, str, bool, str]], float]:
    rows = []
    total_pure = 0
    total_words = 0
    for prompt in PURITY_PROBES:
        text = neural.seeded_generate(prompt, engine)
        pure, words = _purity(text)
        total_pure += pure
        total_words += words
        rows.append(
            (
                "purity",
                repr(prompt),
                words > 0 and pure == words,
                f"{pure}/{words} pure: {text!r}",
            )
        )
    percent = 100.0 * total_pure / max(total_words, 1)
    return rows, percent


def check_determinism(engine) -> tuple[str, str, bool, str]:
    first = neural.seeded_generate(DETERMINISM_PROMPT, engine)
    second = neural.seeded_generate(DETERMINISM_PROMPT, engine)
    ok = bool(first) and first == second
    return ("determinism", repr(DETERMINISM_PROMPT), ok, f"run1={first!r} run2={second!r}")


def main() -> int:
    rows = check_triggers()
    percent = None
    engine = neural.try_load()
    if engine is None:
        rows.append(("purity", "checkpoint", False, f"no checkpoint at {neural.DEFAULT_MODEL_PATH}"))
        rows.append(("determinism", "checkpoint", False, "no checkpoint loaded"))
    else:
        purity_rows, percent = check_purity(engine)
        rows.extend(purity_rows)
        rows.append(check_determinism(engine))

    failed = [row for row in rows if not row[2]]
    print("MeowBench v4 — Feline Depth")
    print("=" * 78)
    print(f"{'section':<14} {'check':<44} {'result':<6}")
    print("-" * 78)
    for section, label, passed, detail in rows:
        line = f"{section:<14.14} {label:<44.44} {'PASS' if passed else 'FAIL':<6}"
        if not passed:
            line += f"  {detail}"
        print(line)
    print("-" * 78)
    if percent is not None:
        print(f"purity score: {percent:.1f}% of neural words in VOCABULARY")
    print(f"OVERALL: {'PASS' if not failed else f'FAIL ({len(failed)} failed)'}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
