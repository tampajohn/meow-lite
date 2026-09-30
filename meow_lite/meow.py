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


class MeowGenerator:
    """Generates deterministic cat meows seeded by the prompt text."""

    def generate(self, prompt: str) -> str:
        seed = int.from_bytes(hashlib.sha256(prompt.encode("utf-8")).digest(), "big")
        rng = random.Random(seed)
        count = rng.randint(3, 12)
        words = [rng.choice(VOCABULARY) for _ in range(count)]
        sentence = " ".join(words)
        sentence = sentence[0].upper() + sentence[1:]
        return sentence + rng.choice(TERMINALS)
