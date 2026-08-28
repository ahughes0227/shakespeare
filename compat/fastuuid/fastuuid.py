"""A pure-Python stand-in for fastuuid, so LiteLLM imports on free-threaded CPython.

fastuuid is a Rust extension with no free-threaded wheel, and `litellm._uuid` imports it
with no fallback of its own ("Always uses fastuuid for performance"). It is a drop-in
reimplementation of the standard library's `uuid`, so this hands back exactly that: the
same names, the same values, without the speed.

See `docs/adr/0006-free-threaded-python-only.md`. Delete this the day fastuuid
publishes a free-threaded wheel.
"""

from __future__ import annotations

from uuid import (
    NAMESPACE_DNS,
    NAMESPACE_OID,
    NAMESPACE_URL,
    NAMESPACE_X500,
    RESERVED_FUTURE,
    RESERVED_MICROSOFT,
    RESERVED_NCS,
    RFC_4122,
    UUID,
    uuid1,
    uuid3,
    uuid4,
    uuid5,
)

__version__ = "0.14.0"

__all__ = [
    "NAMESPACE_DNS",
    "NAMESPACE_OID",
    "NAMESPACE_URL",
    "NAMESPACE_X500",
    "RESERVED_FUTURE",
    "RESERVED_MICROSOFT",
    "RESERVED_NCS",
    "RFC_4122",
    "UUID",
    "uuid1",
    "uuid3",
    "uuid4",
    "uuid4_as_strings_bulk",
    "uuid4_bulk",
    "uuid5",
]


def uuid4_bulk(n: int) -> list[UUID]:
    """fastuuid's batch helper. Nothing in our tree calls it; kept for surface parity."""
    return [uuid4() for _ in range(n)]


def uuid4_as_strings_bulk(n: int) -> list[str]:
    return [str(uuid4()) for _ in range(n)]
