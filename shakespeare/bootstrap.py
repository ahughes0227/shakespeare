"""The single composition root.

The CLI builds a runtime here, and a future GUI will call the same function.  There is
deliberately no second way to assemble the system.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from platformdirs import user_state_dir

from .admission import AdmissionService
from .agent import ModelCapabilityAgent
from .audit import AuditStore
from .capabilities import CapabilityRegistry
from .executor import Executor
from .gateway import Gateway, LiteLLMGateway, ModelProfile, profile_from_environment
from .operators.builtin import build_registry
from .planner import ModelGoalPlanner
from .prompts import PromptStore
from .registry import OperatorRegistry
from .runtime import Runtime
from .telemetry import Exporter, LangSmithExporter, NullExporter, Tracer
from .verifier import Verifier
from .workflows import WorkflowRegistry


def default_state_root() -> Path:
    override = os.environ.get("SHAKESPEARE_HOME")
    return Path(override) if override else Path(user_state_dir("shakespeare"))


def exporters() -> tuple[Exporter, ...]:
    """LangSmith only when deliberately configured.

    Nothing leaves the machine by default, and when it does, only TelemetryEnvelope
    fields are ever sent.
    """
    project = os.environ.get("LANGSMITH_PROJECT")
    if project and os.environ.get("LANGSMITH_API_KEY"):
        return (LangSmithExporter(project),)
    return (NullExporter(),)


@dataclass(frozen=True)
class Services:
    runtime: Runtime
    audit: AuditStore
    operators: OperatorRegistry
    capabilities: CapabilityRegistry
    workflows: WorkflowRegistry
    state_root: Path


def build_runtime(
    *,
    state_root: Path | None = None,
    planner: Any | None = None,
    agents: dict[str, Any] | None = None,
    gateway: Gateway | None = None,
    profile: ModelProfile | None = None,
    run_id: str = "session",
) -> Services:
    root = (state_root or default_state_root()).expanduser()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)

    operators = build_registry()
    capabilities = CapabilityRegistry()
    workflows = WorkflowRegistry(capabilities=capabilities, operators=operators)
    verifier = Verifier(operators)
    audit = AuditStore(root / "audit.sqlite3")
    tracer = Tracer(run_id, exporters())
    prompts = PromptStore()

    if planner is None or agents is None:
        gateway = gateway or LiteLLMGateway()
        profile = profile or profile_from_environment()
        planner = planner or ModelGoalPlanner(
            gateway=gateway, profile=profile, prompts=prompts
        )
        # One agent serves every capability: the capability's own package supplies the
        # catalog and the pinned prompt, so there is nothing per-capability to wire.
        agents = agents or {
            "*": ModelCapabilityAgent(gateway=gateway, profile=profile, prompts=prompts)
        }

    admission = AdmissionService(
        registry=operators,
        audit=audit,
        workspace=root / "candidates",
    )
    runtime = Runtime(
        operators=operators,
        capabilities=capabilities,
        workflows=workflows,
        verifier=verifier,
        executor=Executor(operators, verifier, tracer=tracer),
        planner=planner,
        agents=agents,
        audit=audit,
        workspace_root=root / "runs",
        tracer=tracer,
        prompts=prompts,
        admission=admission,
    )
    return Services(
        runtime=runtime,
        audit=audit,
        operators=operators,
        capabilities=capabilities,
        workflows=workflows,
        state_root=root,
    )
