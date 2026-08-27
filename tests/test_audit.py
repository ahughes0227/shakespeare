from __future__ import annotations

import pytest
from shakespeare.contracts import (
    Composition,
    DomainGoal,
    Invocation,
    ObligationResult,
    StageDecision,
    StagePlan,
    StageVerdict,
    utc_now,
)
from shakespeare.runtime.audit import AuditStore
from shakespeare.runtime.audit.schema import metadata
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError


def _seed_run(store: AuditStore, run_id: str = "r1") -> str:
    store.record_run(
        run_id=run_id,
        workflow_id="rename_files",
        workflow_version="1.0.0",
        workflow_digest="a" * 64,
        request_digest="b" * 64,
        input_root_digest="c" * 64,
    )
    return run_id


class TestAppendOnly:
    @pytest.mark.parametrize("action", ["UPDATE runs SET workflow_id='x'", "DELETE FROM runs"])
    def test_mutation_is_rejected(self, store: AuditStore, action: str) -> None:
        _seed_run(store)
        with pytest.raises(IntegrityError, match="append-only"):
            with store.engine.begin() as connection:
                connection.execute(text(action))

    def test_every_table_has_both_triggers(self, store: AuditStore) -> None:
        with store.engine.begin() as connection:
            names = {
                row[0]
                for row in connection.execute(
                    text("SELECT name FROM sqlite_master WHERE type='trigger'")
                )
            }
        for table in metadata.sorted_tables:
            assert f"{table.name}_no_update" in names
            assert f"{table.name}_no_delete" in names


class TestStageDag:
    def test_records_nodes_and_edges_for_every_attempt(self, store: AuditStore) -> None:
        run_id = _seed_run(store)
        composition = Composition(
            domain_id="content_acquisition",
            invocations=(
                Invocation(invocation_id="scan", operator="fs.scan"),
                Invocation(invocation_id="extract", operator="doc.extract", inputs=("scan",)),
            ),
        )
        plan = StagePlan(
            activated=(
                DomainGoal(
                    domain_id="content_acquisition", goal="get text", success_criterion="all"
                ),
            )
        )
        for attempt_no, decision in ((1, StageDecision.RERUN), (2, StageDecision.ACCEPT)):
            store.record_attempt(
                run_id=run_id,
                stage_name="extract",
                stage_version="1.0.0",
                attempt_no=attempt_no,
                started_at=utc_now().isoformat(),
                plan=plan,
                compositions=[
                    (
                        composition,
                        [
                            {"invocation_id": "scan", "succeeded": True},
                            {"invocation_id": "extract", "succeeded": attempt_no == 2},
                        ],
                    )
                ],
                obligations=[
                    ObligationResult(
                        obligation_id="every_item_has_text_or_reason", passed=attempt_no == 2
                    )
                ],
                verdict=StageVerdict(met=attempt_no == 2, decision=decision),
            )

        dag = store.dag(run_id, "extract")
        assert len(dag["attempts"]) == 2, "failed attempts must remain visible"
        first, second = dag["attempts"]
        assert first["verdict"]["decision"] == "rerun"
        assert second["verdict"]["decision"] == "accept"
        assert len(second["nodes"]) == 2
        assert len(second["edges"]) == 1
        edge = second["edges"][0]
        assert edge["from_invocation"].endswith(":scan")
        assert edge["to_invocation"].endswith(":extract")


class TestCosts:
    def test_aggregates_by_role(self, store: AuditStore) -> None:
        run_id = _seed_run(store)
        for role, cost in (("planner", 0.01), ("planner", 0.02), ("domain", 0.05)):
            store.record_model_invocation(
                run_id=run_id,
                role=role,
                profile_id="p",
                requested_model="openrouter/openai/gpt-5-mini",
                resolved_model="openai/gpt-5-mini",
                provider="openrouter",
                prompt_version="1.0.0",
                cost_usd=cost,
            )
        costs = store.costs(run_id)
        assert costs["model_invocations"] == 3
        assert costs["cost_usd"] == pytest.approx(0.08)
        assert costs["by_role"] == {"planner": pytest.approx(0.03), "domain": pytest.approx(0.05)}
