"""Tiny word-level tokenizer over the meow vocabulary.

Pure-Python subclass of ``transformers.PreTrainedTokenizer`` — no HF
``tokenizers`` dependency. Saves/loads a single ``vocab.json`` next to the
model checkpoint so ``GPT2LMHeadModel.from_pretrained`` +
``MeowTokenizer.from_pretrained`` round-trips a saved directory.
"""

import json
import os

from transformers import PreTrainedTokenizer

from meow_lite.meow import TERMINALS, VOCABULARY

BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"

SPECIAL_TOKENS = [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN]


def build_vocab() -> dict[str, int]:
    tokens: list[str] = []
    for word in VOCABULARY:
        tokens.append(word.lower())
        tokens.append(word.capitalize())
    tokens.extend(TERMINALS)
    tokens.extend(SPECIAL_TOKENS)
    return {token: index for index, token in enumerate(tokens)}


class MeowTokenizer(PreTrainedTokenizer):
    """Word-level tokenizer: meow words (lower + Capitalized), . ! ?, BOS/EOS/PAD."""

    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file=None, **kwargs):
        kwargs.setdefault("bos_token", BOS_TOKEN)
        kwargs.setdefault("eos_token", EOS_TOKEN)
        kwargs.setdefault("pad_token", PAD_TOKEN)
        if vocab_file and os.path.isfile(vocab_file):
            with open(vocab_file, encoding="utf-8") as handle:
                self.vocab = json.load(handle)
        else:
            self.vocab = build_vocab()
        self.id_to_token = {index: token for token, index in self.vocab.items()}
        super().__init__(**kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def _tokenize(self, text, **kwargs):
        tokens = []
        for raw in text.split():
            puncts = []
            while raw and raw[-1] in TERMINALS:
                puncts.append(raw[-1])
                raw = raw[:-1]
            if raw:
                tokens.append(raw)
            tokens.extend(puncts)
        return tokens

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.pad_token_id)

    def _convert_id_to_token(self, index):
        return self.id_to_token.get(index, PAD_TOKEN)

    def _convert_tokens_to_string(self, tokens):
        return self.convert_tokens_to_string(tokens)

    def convert_tokens_to_string(self, tokens):
        pieces: list[str] = []
        for token in tokens:
            if token in TERMINALS:
                if pieces:
                    pieces[-1] = pieces[-1] + token
                else:
                    pieces.append(token)
            else:
                pieces.append(token)
        return " ".join(pieces)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        return [self.bos_token_id] + list(token_ids_0)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        path = os.path.join(save_directory, (filename_prefix or "") + "vocab.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.vocab, handle, ensure_ascii=False, indent=2)
        return (path,)
