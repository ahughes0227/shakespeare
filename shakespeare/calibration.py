"""Is a reported confidence worth anything?

The renderer enforces a floor: a field below it quarantines the file rather than producing
a confidently wrong name. That machinery has been in place and untested since it was
written, because nothing compared a reported 0.8 against how often 0.8 is right. A floor
chosen rather than measured is a guess with a number on it.

This measures it. Given pairs of (what was claimed, what turned out to be true), it reports
where the claims sit against reality and what floor the evidence actually supports. It says
nothing about how a model should decide its confidence — only what its numbers have been
worth so far.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: One claim and its outcome: the confidence reported for a field, and whether the value
#: reported alongside it was right.
Observation = tuple[float, bool]


@dataclass(frozen=True)
class Bucket:
    lower: float
    upper: float
    count: int
    correct: int
    claimed: float

    @property
    def observed(self) -> float:
        """How often claims in this band were actually right."""
        return self.correct / self.count if self.count else 0.0

    @property
    def gap(self) -> float:
        """Claimed minus observed. Positive is overconfidence, which is the dangerous side."""
        return self.claimed - self.observed


def buckets(observations: list[Observation], *, width: float = 0.1) -> tuple[Bucket, ...]:
    """Group claims into confidence bands, keeping only bands that were used."""
    if not 0 < width <= 1:
        raise ValueError("width must fall in (0, 1]")
    count = int(round(1 / width))
    grouped: list[list[Observation]] = [[] for _ in range(count)]
    for confidence, correct in observations:
        index = min(count - 1, max(0, int(confidence / width)))
        grouped[index].append((confidence, correct))
    return tuple(
        Bucket(
            lower=index * width,
            upper=min(1.0, (index + 1) * width),
            count=len(group),
            correct=sum(1 for _, correct in group if correct),
            claimed=sum(confidence for confidence, _ in group) / len(group),
        )
        for index, group in enumerate(grouped)
        if group
    )


def brier(observations: list[Observation]) -> float:
    """Mean squared error of the claims. 0 is perfect, 0.25 is a coin flip claimed at 0.5."""
    if not observations:
        return 0.0
    return sum(
        (confidence - (1.0 if correct else 0.0)) ** 2 for confidence, correct in observations
    ) / len(observations)


def expected_error(observations: list[Observation], *, width: float = 0.1) -> float:
    """How far the claims sit from reality on average, weighted by how often each is made."""
    bands = buckets(observations, width=width)
    total = sum(band.count for band in bands)
    if not total:
        return 0.0
    return sum(band.count * abs(band.gap) for band in bands) / total


def accuracy_above(observations: list[Observation], threshold: float) -> tuple[int, float]:
    """How many claims a floor would accept, and how often those were right."""
    accepted = [correct for confidence, correct in observations if confidence >= threshold]
    if not accepted:
        return 0, 0.0
    return len(accepted), sum(1 for item in accepted if item) / len(accepted)


def floor_for(
    observations: list[Observation], precision: float, *, width: float = 0.05
) -> float | None:
    """The lowest floor whose accepted claims reach `precision`, or None if none does.

    Lowest rather than safest on purpose: every point of floor above what the evidence
    requires quarantines files a person then has to rename by hand, and a floor that
    quarantines everything is precise and useless.
    """
    steps = int(round(1 / width))
    for step in range(steps + 1):
        threshold = round(step * width, 4)
        count, observed = accuracy_above(observations, threshold)
        if count and observed >= precision:
            return threshold
    return None


@dataclass(frozen=True)
class Report:
    observations: int
    fields: dict[str, tuple[int, float]]
    bands: tuple[Bucket, ...]
    brier: float
    expected_error: float
    floors: dict[str, float | None]

    @property
    def overconfident(self) -> bool:
        """True when claims sit above reality overall, which is the direction that hurts."""
        return sum(band.count * band.gap for band in self.bands) > 0


def report(
    per_field: dict[str, list[Observation]], *, targets: tuple[float, ...] = (0.95, 0.99)
) -> Report:
    """Everything measurable about a corpus of claims, per field and overall."""
    combined = [item for values in per_field.values() for item in values]
    return Report(
        observations=len(combined),
        fields={
            name: (len(values), sum(1 for _, ok in values if ok) / len(values))
            for name, values in sorted(per_field.items())
            if values
        },
        bands=buckets(combined),
        brier=brier(combined),
        expected_error=expected_error(combined),
        floors={f"p{int(target * 100)}": floor_for(combined, target) for target in targets},
    )


def observe(
    resolved: list[dict[str, Any]], truth: dict[str, dict[str, str]], *, key: str = "relpath"
) -> dict[str, list[Observation]]:
    """Turn a run's resolved values into claims paired with outcomes.

    `resolved` rows carry the item's identifying key, its `values` and its `confidences`.
    A field the run did not report is not a wrong claim and is not counted — the floor is
    about values that were offered, and an omission is the safe failure working.
    """
    per_field: dict[str, list[Observation]] = {}
    for row in resolved:
        expected = truth.get(str(row.get(key)))
        if expected is None:
            continue
        confidences = row.get("confidences") or {}
        for field, value in (row.get("values") or {}).items():
            if field not in expected or field not in confidences:
                continue
            per_field.setdefault(field, []).append(
                (float(confidences[field]), str(value).strip() == str(expected[field]).strip())
            )
    return per_field
