"""Compile a prompt artifact with DSPy.

Import-time cost is deliberately deferred: `dspy` is an optional extra, and the runtime
must install and run without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from ..contracts import OptimizationRun, PromptArtifact
from ..prompts import PromptStore


class OptimizeError(RuntimeError):
    pass


def require_dspy() -> Any:
    try:
        import dspy
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise OptimizeError(
            "DSPy is an optional extra. Install it with: uv sync --extra optimize"
        ) from exc
    return dspy


@dataclass
class Compiler:
    """Runs an optimizer offline and exports a versioned artifact.

    Never called during a run: optimization is out-of-band by design, so a transactional
    run can never change its own prompts mid-flight.
    """

    store: PromptStore
    optimizer_name: str = "BootstrapFewShot"

    def next_version(self, signature_id: str) -> str:
        versions = self.store.versions(signature_id)
        if not versions:
            return "1.0.0"
        major, minor, _ = (int(part) for part in versions[-1].split("."))
        return f"{major}.{minor + 1}.0"

    def compile(
        self,
        signature_id: str,
        *,
        program: Any,
        trainset: list[Any],
        metric: Any,
        eval_set_digest: str,
        incumbent: PromptArtifact | None,
        incumbent_score: float | None,
    ) -> tuple[PromptArtifact, OptimizationRun]:
        dspy = require_dspy()
        if not trainset:
            raise OptimizeError(
                "no training examples. The audit log needs real runs before optimizing "
                "is worth anything; seed with golden fixtures first."
            )

        optimizer = getattr(dspy.teleprompt, self.optimizer_name)(metric=metric)
        compiled = optimizer.compile(program, trainset=trainset)

        candidate_version = self.next_version(signature_id)
        artifact = PromptArtifact(
            signature_id=signature_id,
            version=candidate_version,
            instructions=_extract_instructions(compiled, fallback=incumbent),
            demonstrations=tuple(_extract_demonstrations(compiled)),
            compiled_from={
                "optimizer": self.optimizer_name,
                "eval_set_digest": eval_set_digest,
                "incumbent_version": incumbent.version if incumbent else None,
            },
        )
        run = OptimizationRun(
            optimization_id=uuid4().hex,
            signature_id=signature_id,
            optimizer=self.optimizer_name,
            eval_set_digest=eval_set_digest,
            incumbent_version=incumbent.version if incumbent else None,
            incumbent_score=incumbent_score,
            candidate_version=candidate_version,
            candidate_score=_score(compiled, trainset, metric),
        )
        return artifact, run


def _extract_instructions(compiled: Any, *, fallback: PromptArtifact | None) -> str:
    for predictor in getattr(compiled, "predictors", lambda: [])():
        signature = getattr(predictor, "signature", None)
        instructions = getattr(signature, "instructions", None)
        if instructions:
            return str(instructions)
    if fallback is not None:
        return fallback.instructions
    raise OptimizeError("the compiled program exposed no instructions to export")


def _extract_demonstrations(compiled: Any) -> list[dict[str, Any]]:
    demonstrations: list[dict[str, Any]] = []
    for predictor in getattr(compiled, "predictors", lambda: [])():
        for demo in getattr(predictor, "demos", []):
            demonstrations.append(dict(demo) if not isinstance(demo, dict) else demo)
    return demonstrations


def _score(program: Any, dataset: list[Any], metric: Any) -> float:
    if not dataset:
        return 0.0
    total = 0.0
    for example in dataset:
        total += float(metric(example, program(**example.inputs()), None))
    return round(total / len(dataset), 6)
