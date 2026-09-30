"""Deterministic prompt-conditional behavior triggers.

Applied above the engine layer: works identically for the neural model and
the v1 rule-based fallback. Same prompt -> same triggers -> same meows.

v4: rules may force MULTIPLE tokens (dog -> <hiss> <stare>; laser/red dot ->
<stare> <pounce>), the reward channel (<purr> + a warm meow mix) leads, and
duplicate tokens across rules are dropped — first occurrence wins.
"""

import re

# Praise: "good cat" / "good boy" / "good girl" / "who's a good" (straight or
# curly apostrophe). Matched against the lowercased prompt.
REWARD_PATTERN = re.compile(r"who['\u2019]s a good\b|\bgood (cat|boy|girl)\b")

TRIGGER_RULES = [
    # (regex applied to the lowercased prompt, tuple of forced action tokens)
    # The reward rule comes FIRST so <purr> always leads.
    (REWARD_PATTERN, ("<purr>",)),  # praise received; motor running
    # v3 rules — unchanged patterns, unchanged relative order
    (re.compile(r"\bbelly\b"), ("<bite>",)),  # touched the belly 0.05s too long
    (re.compile(r"\bcouch|sofa\b"), ("<scratch_couch>",)),
    (re.compile(r"\bglass|table\b"), ("<knock_glass>",)),
    (re.compile(r"\b3 ?am|midnight|night\b"), ("<zoomies>",)),
    (re.compile(r"\bvet\b"), ("<hiss>",)),
    # v4 stimulus triggers
    (re.compile(r"\bcucumbers?\b"), ("<hiss>",)),  # the silent green enemy
    (re.compile(r"\bdogs?\b"), ("<hiss>", "<stare>")),  # order matters
    (re.compile(r"\bbaths?\b|\bwater\b"), ("<zoomies>",)),
    (re.compile(r"\bsprays?\b"), ("<hiss>",)),
    (re.compile(r"\bvacuums?\b"), ("<hiss>",)),
    (re.compile(r"\blasers?\b|\bred dots?\b"), ("<stare>", "<pounce>")),  # order matters
]


def apply_triggers(prompt: str) -> list[str]:
    """Forced action tokens for a prompt, in rule order.

    Each rule fires at most once, contributing its whole token tuple.
    Duplicate tokens across rules are dropped; the first occurrence wins.
    """
    lowered = (prompt or "").lower()
    forced = []
    for pattern, tokens in TRIGGER_RULES:
        if pattern.search(lowered):
            for token in tokens:
                if token not in forced:
                    forced.append(token)
    return forced


def is_reward(prompt: str) -> bool:
    """True for praise prompts — earns a leading <purr> and a warm meow mix."""
    return bool(REWARD_PATTERN.search((prompt or "").lower()))


def weave(forced_tokens: list[str], generated_text: str) -> str:
    """Forced tokens first (in order), then the generated meow text."""
    parts = list(forced_tokens)
    text = (generated_text or "").strip()
    if text:
        parts.append(text)
    return " ".join(parts)
