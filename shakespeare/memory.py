"""What runs have measured, and which declared constant that evidence supports.

Three numbers decide how the system behaves and none of them is measured. `cost_per_item`
is declared in a capability manifest, the confidence floor is declared in a config group,
and `max_goal_attempts` is a field default. Each is a guess with a number on it, and each
is re-guessed identically on every run while the evidence that would settle it is produced,
used once, and discarded at the run boundary.

This module derives the constant the recorded evidence supports. It does not apply it.
Nothing here is read during a run: a measured constant reaches a run by being written into
the manifest or config that declares it, which keeps what a run does fixed and digested at
its start, keeps `replay` a statement about the journal, and leaves the change visible in
git rather than accumulating in a database nobody reviews.

The asymmetry that shapes every threshold here: **underestimating cost is expensive and
overestimating it is cheap.** A batch sized too large is cut off, and that call is billed,
wasted, and retried. A batch sized too small is smaller than it needed to be, and every
call in it still does work. So the estimate leans high, exactly as `plan_batch` does within
a run.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .calibration import Observation, floor_for

#: Below this, a proposal is arithmetic on noise. A live run produces a handful of batches
#: per capability, so this is a few runs' worth rather than a few minutes' worth.
MINIMUM_OBSERVATIONS = 8

#: One run can be strange — an unusual corpus, a bad afternoon at the provider. Two runs
#: agreeing is the cheapest available defence against promoting a fluke.
MINIMUM_RUNS = 2

#: Where in the observed distribution the estimate sits. High on purpose: see the module
#: docstring. Not the maximum, which would let one pathological batch set every later one.
DEFAULT_QUANTILE = 0.8

#: A change smaller than this fraction of the incumbent is not worth a manifest edit.
DEFAULT_MARGIN = 0.1

#: How much the mean item weight may vary across observed batches before one declared
#: per-item number stops being able to describe them. This is ADR 0003's open item — cost
#: is measured per capability, not per item — showing up as a refusal to pretend.
DEFAULT_SPREAD = 3.0


class Verdict(StrEnum):
    """What the evidence says about a proposed constant.

    Deliberately not the promotion outcomes used for prompts: nothing here promotes
    itself, so there is no `auto` case to name. The strongest verdict available is that a
    person should go and write the number down.
    """

    #: The evidence supports the candidate. Pin it.
    SUPPORTED = "supported"
    #: There is enough evidence to have an opinion, and a reason not to act on it alone.
    REVIEW = "review"
    #: Not enough evidence to say anything. Keep the incumbent and keep measuring.
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class Proposal:
    """A constant the evidence supports, and everything needed to disbelieve it."""

    subject: str
    resolved_model: str
    incumbent: float | None
    candidate: float | None
    observations: int
    runs: int
    verdict: Verdict
    rationale: str
    #: Whatever the derivation wants shown alongside the number. Never load-bearing.
    detail: dict[str, Any]

    @property
    def change(self) -> float | None:
        """Candidate over incumbent. 1.0 is no change; 2.0 is twice the estimate."""
        if not self.incumbent or self.candidate is None:
            return None
        return self.candidate / self.incumbent


def quantile(values: list[float], fraction: float) -> float:
    """The value below which `fraction` of the sorted observations fall.

    Linear interpolation between neighbours, so a small sample does not snap the estimate
    to whichever single observation happens to land on the index.
    """
    if not values:
        return 0.0
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must fall in [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def usable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Observations that say something about cost.

    A batch that was cut off did not report what it would have cost, but it proved the
    cost is at least what fitted, so it counts — as `plan_batch` counts it. A batch that
    failed for some other reason says nothing at all about cost: it is evidence about the
    capability, not about the arithmetic, and averaging it in would quietly drag the
    estimate toward whatever a failure happened to spend.
    """
    return [
        row
        for row in rows
        if row.get("weight")
        and row.get("count")
        and (row.get("outcome") or row.get("bound") == "lower")
    ]


def cost_proposal(
    rows: list[dict[str, Any]],
    *,
    incumbent: int | None,
    fraction: float = DEFAULT_QUANTILE,
    margin: float = DEFAULT_MARGIN,
    spread: float = DEFAULT_SPREAD,
    minimum: int = MINIMUM_OBSERVATIONS,
) -> Proposal:
    """The `cost_per_item` the recorded batches support.

    Derived in the unit that survives a change of corpus. What a batch measures is tokens
    per unit of material; `cost_per_item` is that rate declared for an item of average
    weight, which is how `plan_batch` reads it back. So the rate is estimated first and
    converted second, and the average it is converted at is reported, because that average
    is the assumption the number carries.
    """
    subject = rows[0]["subject"] if rows else ""
    model = rows[0]["resolved_model"] if rows else ""
    counted = usable(rows)
    runs = len({row["run_id"] for row in counted})

    def refuse(rationale: str, detail: dict[str, Any] | None = None) -> Proposal:
        return Proposal(
            subject=subject,
            resolved_model=model,
            incumbent=float(incumbent) if incumbent is not None else None,
            candidate=None,
            observations=len(counted),
            runs=runs,
            verdict=Verdict.INSUFFICIENT,
            rationale=rationale,
            detail=detail or {},
        )

    if len({row["resolved_model"] for row in counted}) > 1:
        return refuse(
            "these observations span more than one model, and a measurement taken under "
            "one says nothing about another; filter to one model first"
        )
    if len(counted) < minimum:
        return refuse(
            f"{len(counted)} usable observations, {minimum} needed — "
            f"an estimate from fewer is arithmetic on noise"
        )
    if runs < MINIMUM_RUNS:
        return refuse(
            "every observation comes from the same run — one unusual corpus, or one bad "
            "afternoon at the provider, should not be able to promote itself"
        )

    rates = [row["value"] / row["weight"] for row in counted]
    rate = quantile(rates, fraction)
    material = sum(row["weight"] for row in counted)
    items = sum(row["count"] for row in counted)
    mean_item_weight = material / items
    candidate = int(round(rate * mean_item_weight))

    # The weight per item of each batch, which is what one declared number has to stand in
    # for. A corpus where that varies wildly cannot be described by any single value.
    per_batch = [row["weight"] / row["count"] for row in counted]
    observed_spread = max(per_batch) / min(per_batch) if min(per_batch) > 0 else float("inf")

    detail = {
        "rate": round(rate, 2),
        "mean_item_weight": round(mean_item_weight, 2),
        "item_weight_spread": round(observed_spread, 2),
        "truncations": sum(1 for row in counted if row.get("bound") == "lower"),
        "quantile": fraction,
    }

    def propose(verdict: Verdict, rationale: str) -> Proposal:
        return Proposal(
            subject=subject,
            resolved_model=model,
            incumbent=float(incumbent) if incumbent is not None else None,
            candidate=float(candidate),
            observations=len(counted),
            runs=runs,
            verdict=verdict,
            rationale=rationale,
            detail=detail,
        )

    if incumbent is None:
        return propose(
            Verdict.REVIEW,
            "nothing is declared yet, so there is no incumbent to beat; a first value is "
            "a human decision",
        )
    if observed_spread > spread:
        return propose(
            Verdict.REVIEW,
            f"item weight varies {observed_spread:.1f}x across these batches, so no single "
            f"per-item number describes them — the estimate is sound per unit of material "
            f"and the conversion to one item is the part to distrust",
        )
    if abs(candidate - incumbent) < margin * incumbent:
        # Nothing to do, rather than something to look at: a difference this small is not
        # worth a manifest edit, and it is the same non-answer whether or not it is exact.
        return propose(
            Verdict.INSUFFICIENT,
            f"{candidate} is within {margin:.0%} of the declared {incumbent}; the "
            f"declared value is already what the evidence says",
        )
    direction = "under" if candidate > incumbent else "over"
    return propose(
        Verdict.SUPPORTED,
        f"{len(counted)} observations across {runs} runs put the cost at {candidate}, "
        f"so the declared {incumbent} {direction}states it by "
        f"{abs(candidate - incumbent) / incumbent:.0%}",
    )


def floor_proposal(
    rows: list[dict[str, Any]],
    *,
    incumbent: float | None,
    precision: float = 0.99,
    minimum: int = MINIMUM_OBSERVATIONS,
) -> Proposal:
    """The confidence floor the recorded claims support.

    The floor is the lowest one whose accepted claims reach `precision`, for the reason
    `floor_for` gives: every point above what the evidence requires quarantines a file
    somebody then renames by hand.

    A floor is the one constant here where being wrong in the cheap direction is not
    obviously cheap — too high wastes human time, too low produces a confidently wrong
    name — so this never returns SUPPORTED on its own. Which error to prefer is a
    judgment about the work, not about the data.
    """
    subject = rows[0]["subject"] if rows else "all fields"
    model = rows[0]["resolved_model"] if rows else ""
    runs = len({row["run_id"] for row in rows})
    observations: list[Observation] = [
        (float(row["value"]), bool(row["outcome"])) for row in rows
    ]

    if len(observations) < minimum or runs < MINIMUM_RUNS:
        return Proposal(
            subject=subject,
            resolved_model=model,
            incumbent=incumbent,
            candidate=None,
            observations=len(observations),
            runs=runs,
            verdict=Verdict.INSUFFICIENT,
            rationale=(
                f"{len(observations)} claims across {runs} runs — a floor derived from "
                f"this would describe one afternoon, not the work"
            ),
            detail={},
        )

    candidate = floor_for(observations, precision)
    if candidate is None:
        return Proposal(
            subject=subject,
            resolved_model=model,
            incumbent=incumbent,
            candidate=None,
            observations=len(observations),
            runs=runs,
            verdict=Verdict.REVIEW,
            rationale=(
                f"no floor reaches {precision:.0%} accuracy — at this precision the "
                f"claims are not worth anything, and raising the floor will not fix that"
            ),
            # No accepted count, because no floor was found: reporting one would describe
            # a threshold that does not exist.
            detail={"precision": precision},
        )

    accepted = sum(1 for confidence, _ in observations if confidence >= candidate)
    detail = {
        "precision": precision,
        "accepted": accepted,
        "quarantined": len(observations) - accepted,
    }
    return Proposal(
        subject=subject,
        resolved_model=model,
        incumbent=incumbent,
        candidate=candidate,
        observations=len(observations),
        runs=runs,
        verdict=Verdict.REVIEW,
        rationale=(
            f"{candidate:.2f} is the lowest floor whose accepted claims reach "
            f"{precision:.0%}, keeping {accepted} of {len(observations)} claims; whether "
            f"a wrong name or a hand-renamed file is the worse failure is your call"
        ),
        detail=detail,
    )


@dataclass(frozen=True)
class Recovery:
    """How often a goal that failed was worth another attempt."""

    goal_id: str
    #: attempt number -> (how many times it was reached, how many of those were met)
    by_attempt: dict[int, tuple[int, int]]

    @property
    def deepest_recovery(self) -> int | None:
        """The highest attempt number that ever satisfied a gate."""
        recovered = [number for number, (_, met) in self.by_attempt.items() if met]
        return max(recovered) if recovered else None

    @property
    def wasted(self) -> int:
        """Attempts made beyond the deepest one that ever worked."""
        deepest = self.deepest_recovery or 0
        return sum(reached for number, (reached, _) in self.by_attempt.items()
                   if number > deepest)


def recovery(attempts: dict[str, list[tuple[int, bool]]]) -> tuple[Recovery, ...]:
    """What the log already says about whether a retry was worth making.

    Needs no measurement of its own. `max_goal_attempts` is a chosen number, and whether a
    goal that failed twice ever recovers is a fact the audit log has been recording since
    before anyone asked. ADR 0003 raised the question and left it open; this answers it
    from evidence that already exists.
    """
    result = []
    for goal_id, rows in sorted(attempts.items()):
        by_attempt: dict[int, tuple[int, int]] = {}
        for number, met in rows:
            reached, satisfied = by_attempt.get(number, (0, 0))
            by_attempt[number] = (reached + 1, satisfied + (1 if met else 0))
        result.append(Recovery(goal_id=goal_id, by_attempt=by_attempt))
    return tuple(result)
