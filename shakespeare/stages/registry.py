"""Load versioned stage packages from `_stages/<name>/<version>/`.

Versions sit side by side on disk so two workflows can pin different versions of the
same stage.  That is what makes a stage genuinely reusable: improving one workflow's
stage cannot silently change another's.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ..contracts import SemanticCard, StageSpec

STAGE_ROOT = Path(__file__).resolve().parents[2] / "_stages"


class StageRegistryError(RuntimeError):
    pass


class StageRegistry:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or STAGE_ROOT
        self._specs: dict[str, StageSpec] = {}
        self._cards: dict[str, SemanticCard] = {}
        if self.root.is_dir():
            self._load_all()

    def _load_all(self) -> None:
        for manifest in sorted(self.root.glob("*/*/stage.yml")):
            spec, card = self._load_one(manifest)
            if spec.ref in self._specs:
                raise StageRegistryError(f"duplicate stage package: {spec.ref}")
            self._specs[spec.ref] = spec
            self._cards[spec.ref] = card

    def _load_one(self, manifest: Path) -> tuple[StageSpec, SemanticCard]:
        directory = manifest.parent
        name, version = directory.parent.name, directory.name

        payload = yaml.safe_load(manifest.read_text()) or {}
        spec = StageSpec.model_validate(payload)
        if (spec.name, spec.version) != (name, version):
            raise StageRegistryError(
                f"{manifest}: declares {spec.name}@{spec.version} but lives at {name}/{version}"
            )

        card_path = directory / "stage-context.yml"
        if not card_path.is_file():
            raise StageRegistryError(f"{spec.ref} has no stage-context.yml")
        try:
            card = SemanticCard.model_validate(yaml.safe_load(card_path.read_text()) or {})
        except Exception as exc:
            raise StageRegistryError(
                f"{spec.ref}: stage-context.yml must populate all ten fields ({exc})"
            ) from exc
        return spec, card

    def get(self, ref: str) -> StageSpec:
        try:
            return self._specs[ref]
        except KeyError as exc:
            raise StageRegistryError(
                f"unknown stage: {ref}; registered: {sorted(self._specs)}"
            ) from exc

    def card(self, ref: str) -> SemanticCard:
        return self._cards[ref]

    def refs(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def __contains__(self, ref: object) -> bool:
        return ref in self._specs

    def register(self, spec: StageSpec, card: SemanticCard) -> None:
        """In-memory registration, used by tests to prove the spine is generic."""
        if spec.ref in self._specs:
            raise StageRegistryError(f"duplicate stage package: {spec.ref}")
        self._specs[spec.ref] = spec
        self._cards[spec.ref] = card
