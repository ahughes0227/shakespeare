"""Operator family manifests.

A family is a three-part contract: `family.yml` (revision plus the closed set of
configuration slots), `family-context.yml` (the ten-field semantic card), and a pinned
trusted runner. The first two were written but never read, which meant `allowed_features`
bounded nothing and a family could ship an incomplete card unnoticed.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from .contracts import Contract, OperatorFamily, SemanticCard
from .registry import FAMILY_RUNNERS

TEMPLATE_ROOT = Path(__file__).resolve().parents[1] / "_operator_templates"


class FamilyError(RuntimeError):
    pass


class FamilyManifest(Contract):
    family: OperatorFamily
    revision: str
    #: The closed set of configuration slots this family permits. An answer outside it is
    #: rejected before anything is rendered.
    allowed_features: frozenset[str]

    @property
    def entrypoint(self) -> str:
        return FAMILY_RUNNERS[self.family]


@lru_cache(maxsize=1)
def load_all(root: str | None = None) -> dict[OperatorFamily, tuple[FamilyManifest, SemanticCard]]:
    directory = Path(root) if root else TEMPLATE_ROOT
    loaded: dict[OperatorFamily, tuple[FamilyManifest, SemanticCard]] = {}

    for family in OperatorFamily:
        manifest_path = directory / family / "family.yml"
        if not manifest_path.is_file():
            raise FamilyError(f"{family} has no family.yml at {manifest_path}")
        manifest = FamilyManifest.model_validate(yaml.safe_load(manifest_path.read_text()) or {})
        if manifest.family is not family:
            raise FamilyError(
                f"{manifest_path} declares {manifest.family} but lives in {family}/"
            )

        card_path = directory / family / "family-context.yml"
        if not card_path.is_file():
            raise FamilyError(f"{family} has no family-context.yml")
        try:
            card = SemanticCard.model_validate(yaml.safe_load(card_path.read_text()) or {})
        except Exception as exc:
            raise FamilyError(
                f"{family}: family-context.yml must populate all ten fields ({exc})"
            ) from exc

        loaded[family] = (manifest, card)
    return loaded


def manifest(family: OperatorFamily, root: str | None = None) -> FamilyManifest:
    return load_all(root)[family][0]


def card(family: OperatorFamily, root: str | None = None) -> SemanticCard:
    return load_all(root)[family][1]


def allowed_features(family: OperatorFamily, root: str | None = None) -> frozenset[str]:
    return manifest(family, root).allowed_features


def check_features(family: OperatorFamily, features: frozenset[str]) -> None:
    """Reject configuration slots the family does not declare.

    Without this the `allowed_features` list is decoration: a request could name any slot
    at all and the template would happily render it.
    """
    allowed = allowed_features(family)
    unknown = sorted(features - allowed)
    if unknown:
        raise FamilyError(
            f"{family} does not allow the feature(s) {unknown}; "
            f"declared slots are {sorted(allowed)}"
        )


def verify_marker(package: Path, family: OperatorFamily, root: str | None = None) -> None:
    """Check a rendered package's `{family, revision}` marker.

    The marker is what ties a package to the template that produced it. Rendering one and
    never checking it means a package could claim any family it liked.
    """
    marker_path = package / ".operator-template.yml"
    if not marker_path.is_file():
        raise FamilyError(f"{package} has no .operator-template.yml marker")
    marker = yaml.safe_load(marker_path.read_text()) or {}
    expected = manifest(family, root)
    if marker.get("family") != str(family):
        raise FamilyError(
            f"{package} is marked family {marker.get('family')!r} but was rendered as {family}"
        )
    if str(marker.get("revision")) != expected.revision:
        raise FamilyError(
            f"{package} was rendered from {family} revision {marker.get('revision')!r}, "
            f"but the template is now revision {expected.revision}"
        )
