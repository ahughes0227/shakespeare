# 01 — Runtime

    prompt + input root
      -> planner: workflow_id + RequestContract
      -> workflow spine: stage refs pinned by version
           for each stage, up to max_attempts:
             planner issues StagePlan (activated DomainGoals + SkipDecisions)
             each activated domain subagent compiles a Composition
             runtime validates each Composition and executes it
             obligations checked deterministically (hard gate)
             planner reviews -> StageVerdict: accept | rerun | abort
      -> review stage produces evidence -> planner renders commit verdict
      -> atomic commit from staging, or discard staging

LangGraph owns orchestration, durable checkpointing and `interrupt()` approvals. It does
not make policy. The verifier authorizes; the executor runs; the audit log records.

`runtime.py` is the single composition root. There is no second path.
