"""Deterministic meow generation."""

import hashlib
import random

VOCABULARY = [
    "meow",
    "mew",
    "mewmew",
    "mrow",
    "mrrp",
    "mraow",
    "prrrt",
    "purrr",
    "mrrrow",
    "nyan",
]

TERMINALS = [".", "!", "?"]

# v4 reward channel: praise earns a warmer meow mix, biased toward these.
WARM_WORDS = ["purrr", "prrrt", "mrrp", "mrrrow"]
WARM_WEIGHT = 6


def warm_weights() -> list[int]:
    """Per-word sampling weights over VOCABULARY, biased toward WARM_WORDS."""
    return [WARM_WEIGHT if word in WARM_WORDS else 1 for word in VOCABULARY]


class MeowGenerator:
    """Generates deterministic cat meows seeded by the prompt text."""

    def generate(self, prompt: str, warm: bool = False) -> str:
        """Same prompt (+ same ``warm``) always yields the same sentence.

        ``warm=True`` weights the vocabulary toward WARM_WORDS using the same
        seeded rng; the non-warm path is byte-identical to v1.
        """
        seed = int.from_bytes(hashlib.sha256(prompt.encode("utf-8")).digest(), "big")
        rng = random.Random(seed)
        count = rng.randint(3, 12)
        if warm:
            words = rng.choices(VOCABULARY, weights=warm_weights(), k=count)
        else:
            words = [rng.choice(VOCABULARY) for _ in range(count)]
        sentence = " ".join(words)
        sentence = sentence[0].upper() + sentence[1:]
        return sentence + rng.choice(TERMINALS)
