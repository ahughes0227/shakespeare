"""Golden-fixture canary runs.

Drift detection is only proactive if something actually re-runs known inputs. A canary
case pins an input tree against the decisions it should produce; running it periodically
turns "a provider silently changed behind an alias" from an invisible failure into a
diff.

This is the one place the real model is deliberately exercised on purpose: the whole
point is to notice when the same prompt, over the same files, stops producing the same
answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .contracts import ChangePlan, RequestContract

CANARY_ROOT = Path(__file__).resolve().parents[1] / "_canaries"


class CanaryError(RuntimeError):
    pass


@dataclass(frozen=True)
class CanaryCase:
    name: str
    prompt: str
    inputs: Path
    expected: tuple[tuple[str, str, str | None], ...]

    @property
    def has_expectation(self) -> bool:
        return bool(self.expected)


@dataclass(frozen=True)
class CanaryResult:
    case: CanaryCase
    produced: tuple[tuple[str, str, str | None], ...]
    error: str | None = None

    @property
    def drifted(self) -> bool:
        return self.error is not None or self.produced != self.case.expected

    def diff(self) -> list[tuple[str, str, str]]:
        """Rows of (source, expected, produced) for entries that disagree."""
        expected = {item[0]: item for item in self.case.expected}
        produced = {item[0]: item for item in self.produced}
        rows: list[tuple[str, str, str]] = []
        for source in sorted(set(expected) | set(produced)):
            first, second = expected.get(source), produced.get(source)
            if first != second:
                rows.append(
                    (
                        source,
                        f"{first[1]} → {first[2]}" if first else "—",
                        f"{second[1]} → {second[2]}" if second else "—",
                    )
                )
        return rows


def decisions_of(plan: ChangePlan) -> tuple[tuple[str, str, str | None], ...]:
    """The part of a plan a canary compares: what happened to each file, not run ids."""
    return tuple(
        sorted(
            (entry.source_ref, str(entry.action), getattr(entry, "target_relpath", None))
            for entry in plan.entries
        )
    )


def load_cases(root: Path | None = None) -> tuple[CanaryCase, ...]:
    directory = root or CANARY_ROOT
    if not directory.is_dir():
        return ()

    cases: list[CanaryCase] = []
    for manifest in sorted(directory.glob("*/case.yml")):
        payload = yaml.safe_load(manifest.read_text()) or {}
        prompt = payload.get("prompt")
        if not prompt:
            raise CanaryError(f"{manifest} declares no prompt")

        inputs = manifest.parent / payload.get("inputs", "inputs")
        if not inputs.is_dir():
            raise CanaryError(f"{manifest}: input tree not found at {inputs}")

        expected_path = manifest.parent / "expected.json"
        expected: tuple[tuple[str, str, str | None], ...] = ()
        if expected_path.is_file():
            expected = tuple(
                (item["source"], item["action"], item.get("target"))
                for item in json.loads(expected_path.read_text())
            )
        cases.append(
            CanaryCase(
                name=manifest.parent.name,
                prompt=prompt,
                inputs=inputs,
                expected=tuple(sorted(expected)),
            )
        )
    return tuple(cases)


def record(case: CanaryCase, plan: ChangePlan, root: Path | None = None) -> Path:
    """Capture the current decisions as the expectation.

    Deliberately a separate, explicit action: if a canary re-recorded itself on every
    run it would never detect anything.
    """
    directory = (root or CANARY_ROOT) / case.name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "expected.json"
    path.write_text(
        json.dumps(
            [
                {"source": source, "action": action, "target": target}
                for source, action, target in decisions_of(plan)
            ],
            indent=2,
        )
        + "\n"
    )
    return path


def run_case(runtime: Any, case: CanaryCase, *, output_root: Path) -> CanaryResult:
    """Plan the case without writing anything. A canary never commits."""
    request = RequestContract(
        request_id=f"canary-{case.name}",
        prompt=case.prompt,
        input_root=str(case.inputs.resolve()),
        output_root=str(output_root),
    )
    result = runtime.run(request, commit=False)
    if result.plan is None:
        return CanaryResult(
            case=case, produced=(), error=result.detail or f"run ended as {result.outcome}"
        )
    return CanaryResult(case=case, produced=decisions_of(result.plan))
