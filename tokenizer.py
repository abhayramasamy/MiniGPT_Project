from transformers import AutoTokenizer
import config

_tokenizer = None


def load_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    return _tokenizer


def encode(text, tokenizer=None):
    tok = tokenizer or load_tokenizer()
    return tok.encode(text, add_special_tokens=False)


def decode(ids, tokenizer=None):
    tok = tokenizer or load_tokenizer()
    return tok.decode(ids, skip_special_tokens=True)