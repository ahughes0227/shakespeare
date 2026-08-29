"""A pure-Python stand-in for orjson, so LangGraph runs on free-threaded CPython.

orjson ships no `cp314t` wheel and its build script refuses a free-threaded interpreter
outright, but `langgraph_sdk` imports it eagerly, which puts it on the import path of
`langgraph` and therefore of `system.runtime.durability`. This module supplies the part
of orjson's surface that LangGraph, the LangGraph SDK and LangSmith actually use, on top
of the standard library's `json`.

It aims at orjson's *observable* behaviour, not its speed: `bytes` out, compact
separators, RFC 3339 datetimes, `null` for non-finite floats, non-string keys refused
unless `OPT_NON_STR_KEYS` is set. Where orjson would raise, this raises the same class.

See `docs/adr/0006-free-threaded-python-only.md`. Delete this the day orjson
publishes a free-threaded wheel.
"""

from __future__ import annotations

import dataclasses
import datetime
import enum
import json
import math
import secrets
import uuid
from typing import Any

__version__ = "3.12.0"

__all__ = [
    "Fragment",
    "JSONDecodeError",
    "JSONEncodeError",
    "OPT_APPEND_NEWLINE",
    "OPT_INDENT_2",
    "OPT_NAIVE_UTC",
    "OPT_NON_STR_KEYS",
    "OPT_OMIT_MICROSECONDS",
    "OPT_PASSTHROUGH_DATACLASS",
    "OPT_PASSTHROUGH_DATETIME",
    "OPT_PASSTHROUGH_SUBCLASS",
    "OPT_SERIALIZE_DATACLASS",
    "OPT_SERIALIZE_NUMPY",
    "OPT_SERIALIZE_UUID",
    "OPT_SORT_KEYS",
    "OPT_STRICT_INTEGER",
    "OPT_UTC_Z",
    "dumps",
    "loads",
]

# orjson's own values, because callers or them together and pass the result on.
OPT_INDENT_2 = 1
OPT_NAIVE_UTC = 2
OPT_NON_STR_KEYS = 4
OPT_OMIT_MICROSECONDS = 8
OPT_PASSTHROUGH_DATACLASS = 16
OPT_PASSTHROUGH_DATETIME = 32
OPT_PASSTHROUGH_SUBCLASS = 64
OPT_SERIALIZE_NUMPY = 128
OPT_SORT_KEYS = 256
OPT_STRICT_INTEGER = 512
OPT_UTC_Z = 1024
OPT_APPEND_NEWLINE = 2048
# Deprecated upstream: dataclasses and UUIDs are serialized unconditionally.
OPT_SERIALIZE_DATACLASS = 0
OPT_SERIALIZE_UUID = 0


class JSONEncodeError(TypeError):
    """Raised when a value cannot be serialized. `TypeError`, exactly as upstream."""


class JSONDecodeError(json.JSONDecodeError, ValueError):
    """Raised on malformed input. Subclasses `json.JSONDecodeError`, as upstream does."""


class Fragment:
    """Pre-serialized JSON, spliced into the output verbatim rather than re-encoded."""

    __slots__ = ("contents",)

    def __init__(self, contents: bytes | bytearray | memoryview | str) -> None:
        self.contents = contents


# A dict is deeper than 254 levels only if something is wrong; orjson refuses there too,
# and the limit keeps a cycle from becoming a segfault-shaped RecursionError.
_MAX_DEPTH = 254
_INT_MIN, _INT_MAX = -(2**63), 2**64 - 1


def dumps(obj: Any, /, default: Any = None, option: int | None = None) -> bytes:
    opt = option or 0
    fragments: list[str] = []
    # Fragments cannot survive `json.dumps`, so they leave as a placeholder string and
    # are substituted back afterwards. The nonce is what keeps a placeholder from
    # colliding with a string that happens to appear in the payload.
    nonce = f"\x00orjson-{secrets.token_hex(8)}"
    rendered = _plain(obj, default, opt, fragments, nonce, 0)
    indented = bool(opt & OPT_INDENT_2)
    try:
        text = json.dumps(
            rendered,
            ensure_ascii=False,
            allow_nan=False,
            check_circular=False,
            indent=2 if indented else None,
            separators=(",", ": ") if indented else (",", ":"),
            sort_keys=bool(opt & OPT_SORT_KEYS),
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - _plain has handled these
        raise JSONEncodeError(str(exc)) from exc
    for index, payload in enumerate(fragments):
        text = text.replace(json.dumps(f"{nonce}:{index}"), payload, 1)
    if opt & OPT_APPEND_NEWLINE:
        text += "\n"
    return text.encode("utf-8")


def loads(data: bytes | bytearray | memoryview | str, /) -> Any:
    if isinstance(data, memoryview):
        data = bytes(data)
    try:
        return json.loads(data)
    except json.JSONDecodeError as exc:
        raise JSONDecodeError(exc.msg, exc.doc, exc.pos) from None
    except UnicodeDecodeError as exc:
        raise JSONDecodeError(str(exc), "", 0) from None


def _plain(
    obj: Any, default: Any, opt: int, fragments: list[str], nonce: str, depth: int
) -> Any:
    """Reduce `obj` to something `json.dumps` accepts, applying orjson's rules."""
    if depth > _MAX_DEPTH:
        raise JSONEncodeError("Recursion limit reached")
    if obj is None or isinstance(obj, str):
        return obj
    # bool before int: `isinstance(True, int)` is true, and orjson emits `true`.
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        if not _INT_MIN <= obj <= _INT_MAX:
            raise JSONEncodeError("Integer exceeds 64-bit range")
        if opt & OPT_STRICT_INTEGER and not -(2**53) < obj < 2**53:
            raise JSONEncodeError("Integer exceeds 53-bit range")
        return int(obj)
    if isinstance(obj, float):
        # orjson writes `null` rather than the non-standard `NaN`/`Infinity` literals.
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, Fragment):
        contents = obj.contents
        if isinstance(contents, (bytes, bytearray, memoryview)):
            contents = bytes(contents).decode("utf-8")
        fragments.append(contents)
        return f"{nonce}:{len(fragments) - 1}"
    if isinstance(obj, dict):
        return {
            _key(key, opt): _plain(value, default, opt, fragments, nonce, depth + 1)
            for key, value in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_plain(item, default, opt, fragments, nonce, depth + 1) for item in obj]
    if isinstance(obj, enum.Enum):
        return _plain(obj.value, default, opt, fragments, nonce, depth + 1)
    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        if opt & OPT_PASSTHROUGH_DATETIME:
            return _default(obj, default, opt, fragments, nonce, depth)
        return _timestamp(obj, opt)
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        if opt & OPT_PASSTHROUGH_DATACLASS:
            return _default(obj, default, opt, fragments, nonce, depth)
        fields = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
        return _plain(fields, default, opt, fragments, nonce, depth + 1)
    if opt & OPT_SERIALIZE_NUMPY and type(obj).__module__.partition(".")[0] == "numpy":
        # `tolist` covers both arrays and scalars, and avoids importing numpy here.
        return _plain(obj.tolist(), default, opt, fragments, nonce, depth + 1)
    return _default(obj, default, opt, fragments, nonce, depth)


def _default(
    obj: Any, default: Any, opt: int, fragments: list[str], nonce: str, depth: int
) -> Any:
    if default is None:
        raise JSONEncodeError(f"Type is not JSON serializable: {type(obj).__name__}")
    return _plain(default(obj), None, opt, fragments, nonce, depth + 1)


def _key(key: Any, opt: int) -> str:
    if isinstance(key, str):
        return key
    if not opt & OPT_NON_STR_KEYS:
        raise JSONEncodeError(f"Dict key must be str: {type(key).__name__}")
    if isinstance(key, bool):
        return "true" if key else "false"
    if isinstance(key, (int, float)):
        return json.dumps(key)
    if key is None:
        return "null"
    if isinstance(key, enum.Enum):
        return _key(key.value, opt)
    if isinstance(key, (datetime.datetime, datetime.date, datetime.time)):
        return _timestamp(key, opt)
    if isinstance(key, uuid.UUID):
        return str(key)
    raise JSONEncodeError(f"Dict key is not JSON serializable: {type(key).__name__}")


def _timestamp(value: datetime.datetime | datetime.date | datetime.time, opt: int) -> str:
    if isinstance(value, datetime.date) and not isinstance(value, datetime.datetime):
        return value.isoformat()
    if opt & OPT_OMIT_MICROSECONDS:
        value = value.replace(microsecond=0)
    if (
        opt & OPT_NAIVE_UTC
        and isinstance(value, datetime.datetime)
        and value.tzinfo is None
    ):
        value = value.replace(tzinfo=datetime.UTC)
    rendered = value.isoformat()
    if opt & OPT_UTC_Z and rendered.endswith("+00:00"):
        rendered = f"{rendered[:-6]}Z"
    return rendered
