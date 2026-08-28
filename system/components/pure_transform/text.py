"""Deterministic text normalisation."""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")


def normalize(
    values: dict[str, str],
    *,
    collapse_whitespace: bool = True,
    strip: bool = True,
    aliases: dict[str, str] | None = None,
    case: str = "preserve",
) -> dict[str, str]:
    output: dict[str, str] = {}
    for key, value in values.items():
        text = unicodedata.normalize("NFC", str(value))
        if collapse_whitespace:
            text = _WHITESPACE.sub(" ", text)
        if strip:
            text = text.strip()
        if aliases:
            text = aliases.get(text, text)
        if case == "lower":
            text = text.lower()
        elif case == "upper":
            text = text.upper()
        elif case == "title":
            text = text.title()
        output[key] = text
    return output
