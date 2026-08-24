"""The single composition root.

The CLI builds a runtime here, and a future GUI will call the same function.  There is
deliberately no second way to assemble the system.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_state_dir

from .admission import AdmissionService
from .agent import DomainAgent, ModelDomainAgent
from .audit import AuditStore
from .executor import Executor
from .gateway import Gateway, LiteLLMGateway, ModelProfile, profile_from_environment
from .operators.builtin import build_registry
from .planner import ModelPlanner, Planner
from .prompts import PromptStore
from .registry import OperatorRegistry
from .runtime import Runtime
from .stages import StageRegistry
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
    stages: StageRegistry
    workflows: WorkflowRegistry
    state_root: Path


def build_runtime(
    *,
    state_root: Path | None = None,
    planner: Planner | None = None,
    agents: dict[str, DomainAgent] | None = None,
    gateway: Gateway | None = None,
    profile: ModelProfile | None = None,
    run_id: str = "session",
) -> Services:
    root = (state_root or default_state_root()).expanduser()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)

    operators = build_registry()
    stages = StageRegistry()
    workflows = WorkflowRegistry(stages=stages, operators=operators)
    verifier = Verifier(operators)
    audit = AuditStore(root / "audit.sqlite3")
    tracer = Tracer(run_id, exporters())
    prompts = PromptStore()

    if planner is None or agents is None:
        gateway = gateway or LiteLLMGateway()
        profile = profile or profile_from_environment()
        planner = planner or ModelPlanner(gateway=gateway, profile=profile, prompts=prompts)
        # One agent implementation serves every domain: the domain's own package supplies
        # the catalog and the pinned prompt, so there is nothing per-domain to wire.
        agents = agents or {"*": ModelDomainAgent(gateway=gateway, profile=profile,
                                                  prompts=prompts)}

    admission = AdmissionService(
        registry=operators,
        audit=audit,
        workspace=root / "candidates",
    )
    runtime = Runtime(
        operators=operators,
        stages=stages,
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
        stages=stages,
        workflows=workflows,
        state_root=root,
    )
