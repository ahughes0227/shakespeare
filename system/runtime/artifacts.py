"""Artifacts: the evidence layer.

Per the framework (§11–12), artifacts are not a decomposition level — they are the
persistent evidence that connects capabilities, gates, memory and future reasoning.
Progress is gated on what evidence exists and how good it is, not on which named stage
has finished.

A capability communicates upward through artifacts rather than by exposing what it did
internally. That is what lets a capability reorganise its own work without anything above
it having to care.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import Field

from ..contracts import Contract, content_digest, utc_now


class Quality(StrEnum):
    """How far an artifact goes towards satisfying what was asked of it.

    A gate reads this. `PARTIAL` is the one that matters: it is how a capability says
    "this is correct as far as it goes, and there is more" without failing.
    """

    COMPLETE = "complete"
    PARTIAL = "partial"
    #: Produced, but by a fallback that lost something — OCR where text was wanted.
    DEGRADED = "degraded"
    EMPTY = "empty"


class Artifact(Contract):
    artifact_id: str
    #: What kind of evidence this is, e.g. FileInventory, ExtractedContent, NamingSpec.
    #: Goals require kinds; capabilities declare which they produce.
    kind: str
    run_id: str
    produced_by: str
    #: Digest of the payload. Named distinctly from Contract.digest(), which is the
    #: digest of this descriptor.
    payload_digest: str
    quality: Quality = Quality.COMPLETE
    #: Counts and coverage a gate can check without reading the payload.
    summary: dict[str, Any] = Field(default_factory=dict)
    #: Where the payload lives in the workspace. Bulk content never travels in the
    #: artifact itself, so state and telemetry stay small.
    payload_ref: str | None = None
    created_at: str = ""

    @property
    def usable(self) -> bool:
        return self.quality in (Quality.COMPLETE, Quality.PARTIAL, Quality.DEGRADED)


class ArtifactStoreError(RuntimeError):
    pass


@dataclass
class ArtifactStore:
    """Payloads on disk, descriptors in memory and in the audit log.

    Keeping the payload out of the descriptor is what keeps graph state, prompts and
    telemetry small when the evidence is a hundred megabytes of extracted text.
    """

    root: Path
    run_id: str
    _index: dict[str, Artifact] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def put(
        self,
        *,
        kind: str,
        payload: Any,
        produced_by: str,
        quality: Quality = Quality.COMPLETE,
        summary: dict[str, Any] | None = None,
    ) -> Artifact:
        artifact_id = uuid4().hex
        path = self.root / f"{kind}-{artifact_id}.json"
        path.write_text(json.dumps(payload, default=str))
        artifact = Artifact(
            artifact_id=artifact_id,
            kind=kind,
            run_id=self.run_id,
            produced_by=produced_by,
            quality=quality,
            payload_digest=content_digest(payload),
            summary=summary or {},
            payload_ref=str(path),
            created_at=utc_now().isoformat(),
        )
        self._index[artifact_id] = artifact
        return artifact

    def load(self, artifact: Artifact) -> Any:
        if artifact.payload_ref is None:
            raise ArtifactStoreError(f"{artifact.kind} {artifact.artifact_id} has no payload")
        return json.loads(Path(artifact.payload_ref).read_text())

    def of_kind(self, kind: str) -> tuple[Artifact, ...]:
        return tuple(
            item for item in self._index.values() if item.kind == kind and item.usable
        )

    def latest(self, kind: str) -> Artifact | None:
        matching = sorted(self.of_kind(kind), key=lambda item: item.created_at)
        return matching[-1] if matching else None

    def payload_of(self, kind: str) -> Any | None:
        artifact = self.latest(kind)
        return self.load(artifact) if artifact else None

    def kinds(self) -> frozenset[str]:
        return frozenset(item.kind for item in self._index.values() if item.usable)

    def all(self) -> tuple[Artifact, ...]:
        return tuple(sorted(self._index.values(), key=lambda item: item.created_at))

    def describe(self) -> list[dict[str, Any]]:
        """What a planner or a gate is shown: evidence and its quality, never payloads."""
        return [
            {
                "kind": item.kind,
                "quality": str(item.quality),
                "produced_by": item.produced_by,
                "summary": item.summary,
                "digest": item.payload_digest[:12],
            }
            for item in self.all()
        ]
