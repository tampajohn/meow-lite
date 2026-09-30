"""Optional neural meow generation: tiny from-scratch GPT-2 over the meow vocab.

If a trained checkpoint exists at MEOW_LITE_MODEL_PATH (default
``models/meow-lite``), generation is neural with sha256(prompt)-seeded
determinism. Any load failure falls back silently to the v1 MeowGenerator.
"""

import hashlib
import os
import random
from pathlib import Path

DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "meow-lite")
MAX_NEW_TOKENS = 16
TEMPERATURE = 1.0


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


def seeded_generate(prompt: str, engine) -> str:
    """Deterministic sampling: same prompt -> same meows."""
    import numpy as np
    import torch

    model = engine["model"]
    tokenizer = engine["tokenizer"]
    seed = int.from_bytes(hashlib.sha256(prompt.encode("utf-8")).digest(), "big")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed % (2**63 - 1))

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
        )
    return tokenizer.decode(output[0][1:], skip_special_tokens=True).strip()
