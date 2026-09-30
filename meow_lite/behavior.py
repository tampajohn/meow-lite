"""Deterministic prompt-conditional misbehavior triggers.

Applied above the engine layer: works identically for the neural model and
the v1 rule-based fallback. Same prompt -> same triggers -> same meows.
"""

import re

TRIGGER_RULES = [
    # (regex applied to the lowercased prompt, forced action token)
    (re.compile(r"\bbelly\b"), "<bite>"),  # touched the belly 0.05s too long
    (re.compile(r"\bcouch|sofa\b"), "<scratch_couch>"),
    (re.compile(r"\bglass|table\b"), "<knock_glass>"),
    (re.compile(r"\b3 ?am|midnight|night\b"), "<zoomies>"),
    (re.compile(r"\bvet\b"), "<hiss>"),
]


def apply_triggers(prompt: str) -> list[str]:
    """Forced action tokens for a prompt, in rule order, max one per rule."""
    lowered = (prompt or "").lower()
    forced = []
    for pattern, token in TRIGGER_RULES:
        if pattern.search(lowered):
            forced.append(token)
    return forced


def weave(forced_tokens: list[str], generated_text: str) -> str:
    """Forced tokens first (in order), then the generated meow text."""
    parts = list(forced_tokens)
    text = (generated_text or "").strip()
    if text:
        parts.append(text)
    return " ".join(parts)
