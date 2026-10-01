"""v6 neural engine: from-scratch GPT-2 that reads English, speaks cat only.

Output constraint (specs/v6.md): after [SEP], a LogitsProcessor masks every
token except the 10 cat action tokens, the meow vocabulary (both cases, from
meow_lite.meow lists), '.', '!', '?', and [EOS]. English leakage is
impossible by construction.

Determinism: sha256(prompt) seeds random/numpy/torch (same scheme as
neural.py); sampling at temperature 0.4 / top_p 0.95.
"""

import hashlib
import random
import re
from pathlib import Path

DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "meow-lite-v6")
MAX_NEW_TOKENS = 48  # BPE splits meow words into subwords; v4's 3-12 words ≈ 8-30 tokens
MIN_NEW_TOKENS = 14  # ~4-5 word floor: "Purrr." is only 6 BPE subwords; v4-grade chattiness

_PUNCT_RUN = re.compile(r"([.!?]){2,}")
_GLUE_AFTER_PUNCT = re.compile(r"([.!?])([^\s])")


def _polish(text: str) -> str:
    """Presentation cleanup so v6 prose reads like v4: spaces between glued
    action tokens and words, single punctuation marks, capitalized first letter."""
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("><", "> <")  # "<hiss><stare>" -> "<hiss> <stare>"
    text = re.sub(r">(?=[A-Za-z])", "> ", text)  # "<stare>Mraow" -> "<stare> Mraow"
    text = re.sub(r"([A-Za-z.])(?=<)", r"\1 ", text)  # "Mraow<bite>" -> "Mraow <bite>"
    text = _PUNCT_RUN.sub(r"\1", text)  # "?!?!" -> "?"
    text = _GLUE_AFTER_PUNCT.sub(r"\1 \2", text)  # "Mrow!Prrrt" -> "Mrow! Prrrt"
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)  # BPE camelGlue -> two words
    text = re.sub(r"\s+", " ", text).strip()
    if not text.startswith("<"):
        match = re.search(r"[A-Za-z]", text)
        if match:
            i = match.start()
            text = text[:i] + text[i].upper() + text[i + 1 :]
    return text
TEMPERATURE = 0.4
TOP_P = 0.95

from meow_lite.meow import TERMINALS, VOCABULARY, WARM_WORDS  # noqa: E402
from meow_lite.tokenizer import ACTION_TOKENS  # noqa: E402
from meow_lite import tokenizer_v6  # noqa: E402


def cat_token_ids(tokenizer: "tokenizer_v6.TokenizerV6") -> list[int]:
    """Every token id the model may emit after [SEP].

    The 10 action tokens, the meow/warm vocabulary in both cases (encoded by
    the v6 BPE — a word may span several ids, all become legal), terminals,
    and [EOS]. Computed once at load.
    """
    words = set(VOCABULARY) | set(WARM_WORDS)
    forms = []
    for word in words:
        forms.append(word.lower())
        forms.append(word.capitalize())
    allowed = set()
    for form in forms:
        allowed.update(tokenizer.encode_ids(form))
    for terminal in TERMINALS:
        allowed.update(tokenizer.encode_ids(terminal))
    allowed.update(tokenizer.action_ids)
    allowed.add(tokenizer.eos_id)
    return sorted(allowed)


def build_cat_mask(tokenizer: "tokenizer_v6.TokenizerV6", model):
    """CatMaskLogitsProcessor bound to this tokenizer/model."""
    from transformers import LogitsProcessor

    allowed = cat_token_ids(tokenizer)
    vocab_size = model.config.vocab_size

    class CatMaskLogitsProcessor(LogitsProcessor):
        """After [SEP], every non-cat token id is masked to -inf."""

        def __init__(self):
            self.sep_token_id = tokenizer.sep_id
            self.allowed = allowed
            self._mask = None
            self._vocab_size = vocab_size

        def _mask_for(self, device, dtype):
            import torch

            if self._mask is None or self._mask.device != device:
                full = torch.full((self._vocab_size,), float("-inf"), dtype=dtype)
                full[list(self.allowed)] = 0.0
                self._mask = full.to(device)
            return self._mask

        def __call__(self, input_ids, scores):
            import torch

            if not (input_ids == self.sep_token_id).any():
                return scores
            mask = self._mask_for(scores.device, scores.dtype)
            return scores + mask

    return CatMaskLogitsProcessor()


def build_action_once(tokenizer: "tokenizer_v6.TokenizerV6"):
    """ActionOnceLogitsProcessor: each action token may fire ONCE per response.

    Kills degenerate '<bite> <bite> <bite>' cascades (the decision-weighted
    model loops action tokens at low temperature) while keeping tuples like
    '<hiss> <stare>' legal. Meow prose words are unaffected.
    """
    from transformers import LogitsProcessor

    action_ids = set(tokenizer.action_ids)

    class ActionOnceLogitsProcessor(LogitsProcessor):
        def __init__(self):
            self.sep_id = tokenizer.sep_id

        def __call__(self, input_ids, scores):
            seq = input_ids[0].tolist()
            sep_pos = len(seq) - 1 - seq[::-1].index(self.sep_id) if self.sep_id in seq else 0
            seen = set(seq[sep_pos:]) & action_ids
            if seen:
                scores[:, list(seen)] = float("-inf")
            return scores

    return ActionOnceLogitsProcessor()


def load(model_dir=None):
    """Load the v6 checkpoint; returns (model, tokenizer) dict or None."""
    path = model_dir or DEFAULT_MODEL_PATH
    if not Path(path).is_dir():
        return None
    try:
        import torch
        from transformers import GPT2LMHeadModel

        model = GPT2LMHeadModel.from_pretrained(path)
        model.eval()
        tokenizer = tokenizer_v6.load(path)
        processor = build_cat_mask(tokenizer, model)
        return {
            "model": model,
            "tokenizer": tokenizer,
            "path": str(path),
            "cat_mask": processor,
            "action_once": build_action_once(tokenizer),
        }
    except Exception:
        return None


def seeded_generate(prompt: str, engine, max_new: int = MAX_NEW_TOKENS) -> str:
    """Deterministic cat-only generation: same prompt -> byte-identical meows."""
    import numpy as np
    import torch

    model = engine["model"]
    tokenizer = engine["tokenizer"]
    seed = int.from_bytes(hashlib.sha256(prompt.encode("utf-8")).digest(), "big")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed % (2**63 - 1))

    input_ids = torch.tensor([tokenizer.prompt_ids(prompt)], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            logits_processor=[engine["cat_mask"], engine["action_once"]],
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_new_tokens=max_new,
            min_new_tokens=min(MIN_NEW_TOKENS, max_new),
            eos_token_id=tokenizer.eos_id,
            pad_token_id=tokenizer.pad_id,
        )
    new_ids = output[0][input_ids.shape[1]:]
    return _polish(tokenizer.decode_response(new_ids))
