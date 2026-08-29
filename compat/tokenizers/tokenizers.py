"""The fallback `tokenizers`, for platforms `vendor/`'s real wheel does not cover.

`tokenizers` publishes abi3 wheels only, and abi3 does not exist for a free-threaded
interpreter, so there is no wheel on PyPI to install and the sdist fails to link.
`vendor/build-tokenizers.sh` builds a real one — upstream's Rust is already free-threaded,
only its packaging is not — but that wheel is macOS arm64 on `cp314t` and nothing else.
Everywhere else this module is what lets `litellm.utils`, which imports `Tokenizer` at
module scope, be imported at all.

LiteLLM reaches for a HuggingFace tokenizer only when counting tokens for a model whose
tokenizer is not tiktoken's — Claude, Cohere and Llama. Those counts cannot be
approximated honestly, so this raises rather than inventing a number, and
`profile_from_environment` turns that refusal into one the caller actually sees: LiteLLM
itself catches it and falls back to tiktoken. OpenAI-shaped models, which is what
`SHAKESPEARE_MODEL` pins, go through tiktoken and never touch this.

See `docs/adr/0006-free-threaded-python-only.md`. Delete this the day tokenizers publishes
a free-threaded wheel of its own.
"""

from __future__ import annotations

from typing import Any, NoReturn

__version__ = "0.23.1"

__all__ = ["Encoding", "Tokenizer", "TokenizerUnavailable"]

_MESSAGE = (
    "tokenizers has no free-threaded build, so HuggingFace token counting is "
    "unavailable on this interpreter. Models whose tokenizer is not tiktoken's "
    "(Claude, Cohere, Llama) cannot be counted or costed here — pin an OpenAI-shaped "
    "model, or see docs/adr/0006-free-threaded-python-only.md."
)


class TokenizerUnavailable(RuntimeError):
    """Raised where a real tokenizer would have been used. Loud on purpose."""

    def __init__(self) -> None:
        super().__init__(_MESSAGE)


def _unavailable(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise TokenizerUnavailable


class Encoding:
    """The shape LiteLLM reads back from `encode`. Never constructed here."""

    ids: list[int]
    tokens: list[str]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _unavailable()


class Tokenizer:
    """`tokenizers.Tokenizer`, minus the tokenizer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _unavailable()

    from_pretrained = staticmethod(_unavailable)
    from_str = staticmethod(_unavailable)
    from_file = staticmethod(_unavailable)
    from_buffer = staticmethod(_unavailable)

    encode = _unavailable
    encode_batch = _unavailable
    decode = _unavailable
    decode_batch = _unavailable
