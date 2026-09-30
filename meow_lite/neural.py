"""Optional neural meow generation: tiny from-scratch GPT-2 over the meow vocab.

If a trained checkpoint exists at MEOW_LITE_MODEL_PATH (default
``models/meow-lite``), generation is neural with sha256(prompt)-seeded
determinism. Any load failure falls back silently to the v1 MeowGenerator.

v4: ``seeded_generate(..., warm=True)`` adds an additive logit bias toward the
warm meow words (the reward channel) via a transformers LogitsProcessor.
Seeding is untouched, so determinism holds: same prompt -> same meows.
"""

import hashlib
import os
import random
from pathlib import Path

from transformers import LogitsProcessor, LogitsProcessorList

from meow_lite.meow import WARM_WORDS

DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "meow-lite")
MAX_NEW_TOKENS = 16
TEMPERATURE = 1.0

# v4 reward channel: additive logit bonus for warm meow words when warm=True.
WARM_BIAS = 8.0


class WarmLogitsBias(LogitsProcessor):
    """Additive logit bias toward warm meow token ids (pure; consumes no RNG)."""

    def __init__(self, token_ids, bias: float = WARM_BIAS):
        self.token_ids = list(token_ids)
        self.bias = float(bias)

    def __call__(self, input_ids, scores):
        scores[:, self.token_ids] += self.bias
        return scores


def warm_token_ids(tokenizer) -> list[int]:
    """Token ids of WARM_WORDS (lower + Capitalized variants)."""
    variants = [variant for word in WARM_WORDS for variant in (word, word.capitalize())]
    return list(tokenizer.convert_tokens_to_ids(variants))


def model_path_from_env() -> str:
    return os.environ.get("MEOW_LITE_MODEL_PATH", DEFAULT_MODEL_PATH)


def try_load(path=None):
    """Load model + tokenizer from ``path``; return dict or None on any failure."""
    path = path or model_path_from_env()
    if not os.path.isdir(path):
        return None
    try:
        import torch
        from transformers import GPT2LMHeadModel

        from meow_lite.tokenizer import MeowTokenizer

        model = GPT2LMHeadModel.from_pretrained(path)
        model.eval()
        tokenizer = MeowTokenizer.from_pretrained(path)
        return {"model": model, "tokenizer": tokenizer, "path": path}
    except Exception:
        return None


def seeded_generate(prompt: str, engine, warm: bool = False) -> str:
    """Deterministic sampling: same prompt (+ same ``warm``) -> same meows.

    ``warm=True`` adds a +WARM_BIAS logit bonus to the warm meow words. The
    bias is a deterministic function of the request applied after seeding —
    it consumes no RNG, so reproducibility is untouched.
    """
    import numpy as np
    import torch

    model = engine["model"]
    tokenizer = engine["tokenizer"]
    seed = int.from_bytes(hashlib.sha256(prompt.encode("utf-8")).digest(), "big")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed % (2**63 - 1))

    logits_processor = None
    if warm:
        logits_processor = LogitsProcessorList(
            [WarmLogitsBias(warm_token_ids(tokenizer))]
        )

    input_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=TEMPERATURE,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            logits_processor=logits_processor,
        )
    return tokenizer.decode(output[0][1:], skip_special_tokens=True).strip()
