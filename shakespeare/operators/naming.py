"""Deterministic name rendering and collision resolution.

This is where consistency actually comes from.  No agent chooses a filename: an agent
supplies field values, and this module renders them through a frozen spec.  Identical
values therefore always produce an identical name, whatever route the agent took to find
them.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from ..contracts import Contract, content_digest

#: Illegal on Windows, and `/` on POSIX.  We apply the union so a plan renders identically
#: on every platform, which matters because a plan is portable data.
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WHITESPACE = re.compile(r"\s+")
#: Windows refuses these stems regardless of extension.
_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)
_PLACEHOLDER = re.compile(r"\{([a-z_][a-z0-9_]*)(?::([^{}]+))?\}")

#: Filled by the renderer from the deterministic scan order, never by a model.
SEQUENCE_FIELD = "seq"


class FieldKind(StrEnum):
    STRING = "string"
    DATE = "date"
    INTEGER = "integer"
    CURRENCY = "currency"
    SEQUENCE = "sequence"


class CasePolicy(StrEnum):
    PRESERVE = "preserve"
    LOWER = "lower"
    UPPER = "upper"
    TITLE = "title"


class FieldDecl(Contract):
    name: str = Field(min_length=1)
    kind: FieldKind = FieldKind.STRING
    #: strftime for dates, a printf-style width for integers, e.g. "04d".
    format: str | None = None
    required: bool = True
    confidence_floor: float = Field(default=0.7, ge=0, le=1)


class NamePolicy(Contract):
    separator: str = ", "
    case: CasePolicy = CasePolicy.PRESERVE
    max_length: int = Field(default=200, ge=16, le=255)
    replacement: str = "-"
    collapse_whitespace: bool = True
    #: Applied after formatting; maps a raw value to its canonical form.
    aliases: dict[str, str] = Field(default_factory=dict)


class RenderResult(Contract):
    item_id: str
    rendered: str | None
    reason: str | None = None

    @property
    def resolved(self) -> bool:
        return self.rendered is not None


def sanitize(value: str, policy: NamePolicy) -> str:
    """Make a fragment safe on every platform, deterministically."""
    text = unicodedata.normalize("NFC", value)
    if policy.collapse_whitespace:
        # Before the illegal sweep: tabs and newlines are *both* whitespace and control
        # characters, and collapsing them first avoids turning them into replacements.
        text = _WHITESPACE.sub(" ", text)
    text = _ILLEGAL.sub(policy.replacement, text)
    if policy.collapse_whitespace:
        text = _WHITESPACE.sub(" ", text)
    return text.strip(" .")


def tidy_separators(stem: str, separator: str) -> str:
    """Remove separators left dangling by an absent optional field.

    An optional field renders as an empty string, which would otherwise leave `a, , b` or
    a trailing `a, `.  Only runs of two or more are collapsed, so a comma inside a real
    value such as `Smith, Jones Ltd` survives.
    """
    token = separator.strip()
    if not token:
        return stem.strip()
    run = rf"(?:\s*{re.escape(token)}\s*)"
    stem = re.sub(rf"{run}{{2,}}", separator, stem)
    stem = re.sub(rf"^{run}+", "", stem)
    stem = re.sub(rf"{run}+$", "", stem)
    return stem.strip()


def _format_value(value: Any, decl: FieldDecl) -> str:
    if decl.kind is FieldKind.DATE:
        moment = value
        if isinstance(moment, str):
            moment = datetime.fromisoformat(moment)
        if isinstance(moment, datetime):
            moment = moment.date()
        if not isinstance(moment, date):
            raise ValueError(f"{decl.name}: expected a date, got {type(value).__name__}")
        return moment.strftime(decl.format or "%Y%m")
    if decl.kind in (FieldKind.INTEGER, FieldKind.SEQUENCE):
        number = int(value)
        return format(number, decl.format or "d")
    if decl.kind is FieldKind.CURRENCY:
        return f"{float(value):.2f}"
    return str(value)


def _apply_case(text: str, policy: CasePolicy) -> str:
    if policy is CasePolicy.LOWER:
        return text.lower()
    if policy is CasePolicy.UPPER:
        return text.upper()
    if policy is CasePolicy.TITLE:
        return text.title()
    return text


def _clamp(stem: str, extension: str, max_length: int) -> str:
    budget = max_length - len(extension)
    if budget <= 0:
        raise ValueError("max_length leaves no room for the extension")
    if len(stem) <= budget:
        return stem
    # Truncate the stem rather than the extension: the extension carries the file type,
    # and changing it would change what the file *is*.
    return stem[:budget].rstrip(" .-")


def render(
    *,
    item_id: str,
    template: str,
    fields: tuple[FieldDecl, ...],
    values: dict[str, Any],
    policy: NamePolicy,
    extension: str = "",
    sequence: int | None = None,
) -> RenderResult:
    """Render one filename.  Total: it never raises for missing data, it explains."""
    declared = {decl.name: decl for decl in fields}
    placeholders = _PLACEHOLDER.findall(template)
    if not placeholders:
        return RenderResult(item_id=item_id, rendered=None, reason="template_has_no_fields")

    for name, _ in placeholders:
        if name not in declared and name != SEQUENCE_FIELD:
            return RenderResult(item_id=item_id, rendered=None, reason=f"undeclared_field:{name}")

    resolved: dict[str, str] = {}
    for name, inline_format in placeholders:
        if name == SEQUENCE_FIELD and name not in declared:
            if sequence is None:
                return RenderResult(item_id=item_id, rendered=None, reason="missing_sequence")
            resolved[name] = format(sequence, inline_format or "d")
            continue

        decl = declared[name]
        if inline_format:
            decl = decl.model_copy(update={"format": inline_format})

        raw = sequence if decl.kind is FieldKind.SEQUENCE else values.get(name)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            if decl.required:
                return RenderResult(item_id=item_id, rendered=None, reason=f"missing_field:{name}")
            resolved[name] = ""
            continue
        try:
            formatted = _format_value(raw, decl)
        except (ValueError, TypeError) as exc:
            return RenderResult(
                item_id=item_id, rendered=None, reason=f"unformattable_field:{name}:{exc}"
            )
        formatted = policy.aliases.get(formatted, formatted)
        resolved[name] = sanitize(formatted, policy)

    def substitute(match: re.Match[str]) -> str:
        return resolved[match.group(1)]

    stem = _PLACEHOLDER.sub(substitute, template)
    stem = tidy_separators(stem, policy.separator)
    stem = sanitize(stem, policy)
    stem = _apply_case(stem, policy.case)
    if not stem:
        return RenderResult(item_id=item_id, rendered=None, reason="rendered_empty")
    if stem.split(".")[0].upper() in _RESERVED:
        stem = f"{policy.replacement}{stem}"
    stem = _clamp(stem, extension, policy.max_length)
    if not stem:
        return RenderResult(item_id=item_id, rendered=None, reason="rendered_empty")
    return RenderResult(item_id=item_id, rendered=f"{stem}{extension}")


# --------------------------------------------------------------------------------------
# Collision resolution
# --------------------------------------------------------------------------------------


class CollisionPolicy(StrEnum):
    SUFFIX_N = "suffix_n"
    HASH_SUFFIX = "hash_suffix"
    FAIL = "fail"


class Candidate(Contract):
    item_id: str
    directory: str
    name: str


class Resolution(Contract):
    item_id: str
    directory: str
    name: str | None
    reason: str | None = None


def _split_extension(name: str) -> tuple[str, str]:
    index = name.rfind(".")
    if index <= 0:
        return name, ""
    return name[:index], name[index:]


def resolve_collisions(
    candidates: tuple[Candidate, ...], policy: CollisionPolicy
) -> tuple[Resolution, ...]:
    """Resolve duplicate targets deterministically.

    Ordering is by (directory, name, item_id) rather than by arrival, so a shuffled input
    produces byte-identical output.  Without that, two runs over the same files could
    disagree about which one keeps the unsuffixed name.
    """
    ordered = sorted(candidates, key=lambda item: (item.directory, item.name, item.item_id))
    taken: set[tuple[str, str]] = set()
    resolutions: list[Resolution] = []

    for candidate in ordered:
        key = (candidate.directory, candidate.name.lower())
        if key not in taken:
            taken.add(key)
            resolutions.append(
                Resolution(
                    item_id=candidate.item_id,
                    directory=candidate.directory,
                    name=candidate.name,
                )
            )
            continue

        if policy is CollisionPolicy.FAIL:
            resolutions.append(
                Resolution(
                    item_id=candidate.item_id,
                    directory=candidate.directory,
                    name=None,
                    reason=f"collision:{candidate.name}",
                )
            )
            continue

        stem, extension = _split_extension(candidate.name)
        if policy is CollisionPolicy.HASH_SUFFIX:
            suffix = content_digest(candidate.item_id)[:8]
            resolved = f"{stem}-{suffix}{extension}"
        else:
            counter = 2
            resolved = f"{stem} ({counter}){extension}"
            while (candidate.directory, resolved.lower()) in taken:
                counter += 1
                resolved = f"{stem} ({counter}){extension}"

        taken.add((candidate.directory, resolved.lower()))
        resolutions.append(
            Resolution(
                item_id=candidate.item_id, directory=candidate.directory, name=resolved
            )
        )

    return tuple(sorted(resolutions, key=lambda item: item.item_id))
