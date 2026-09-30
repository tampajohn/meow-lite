"""v6 tokenizer wrapper: byte-level BPE (HF `tokenizers`) for the v6 model.

The tokenizer is trained in ``tools/train_v6.py`` and saved as
``tokenizer.json`` next to the model checkpoint. This wrapper loads it and
exposes encode/decode plus the structural special ids and action-token ids.

Special tokens: the 10 cat action tokens (never split — they are added
tokens) plus the structural ``[BOS] [SEP] [EOS] [PAD]``.

Decode note: action tokens ARE special added tokens, so plain
``skip_special_tokens=True`` would erase them from responses. Decoding here
keeps them and strips only the structural specials.
"""

import re
from pathlib import Path

from tokenizers import Tokenizer

from meow_lite.tokenizer import ACTION_TOKENS

BOS_TOKEN = "[BOS]"
SEP_TOKEN = "[SEP]"
EOS_TOKEN = "[EOS]"
PAD_TOKEN = "[PAD]"

STRUCTURAL_TOKENS = [BOS_TOKEN, SEP_TOKEN, EOS_TOKEN, PAD_TOKEN]
# Order matters: action tokens first, then structural — this is the order
# they are registered by the BPE trainer in tools/train_v6.py.
SPECIAL_TOKENS = list(ACTION_TOKENS) + STRUCTURAL_TOKENS

_STRUCTURAL_RE = re.compile(r"\[(?:BOS|SEP|EOS|PAD)\]\s*")


class TokenizerV6:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.bos_id = tokenizer.token_to_id(BOS_TOKEN)
        self.sep_id = tokenizer.token_to_id(SEP_TOKEN)
        self.eos_id = tokenizer.token_to_id(EOS_TOKEN)
        self.pad_id = tokenizer.token_to_id(PAD_TOKEN)
        self.action_ids = [tokenizer.token_to_id(t) for t in ACTION_TOKENS]

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size(with_added_tokens=True)

    def token_to_id(self, token: str):
        return self.tokenizer.token_to_id(token)

    def encode(self, text: str):
        """Encode raw text; no structural specials are added automatically."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def encode_ids(self, text: str) -> list[int]:
        return self.encode(text).ids

    def prompt_ids(self, prompt: str) -> list[int]:
        """[BOS] prompt [SEP] — the generation input (model continues after [SEP])."""
        return [self.bos_id] + self.encode_ids(prompt) + [self.sep_id]

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def decode_response(self, ids) -> str:
        """Decode generated ids, keeping action tokens, dropping [EOS]/[PAD] etc."""
        text = self.tokenizer.decode(list(ids), skip_special_tokens=False)
        text = _STRUCTURAL_RE.sub("", text)
        return text.strip()


def load(model_dir) -> TokenizerV6:
    """Load tokenizer.json from a checkpoint dir. Raises if missing."""
    path = Path(model_dir) / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(f"no tokenizer.json under {model_dir}")
    return TokenizerV6(Tokenizer.from_file(str(path)))
