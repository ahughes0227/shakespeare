"""The record store, and what having one changes.

ADR 0004: a model's reading had to survive inside the response that reported it, so a
sixty-invoice run needed batch sizing at all. A record written the moment it is read
needs no transport.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from shakespeare.contracts import BudgetEnvelope, Composition, DomainSpec, Invocation
from shakespeare.executor import Budget, Executor
from shakespeare.operators import records
from shakespeare.operators.builtin import build_registry
from shakespeare.verifier import Verifier


def row(item_id: str, vendor: str = "ACME", confidence: float = 0.9) -> dict:
    return {
        "item_id": item_id,
        "directory": "2024/q1",
        "extension": ".pdf",
        "values": {"vendor": vendor},
        "confidences": {"vendor": confidence},
    }


class TestTheStore:
    def test_what_is_written_can_be_read_back(self, tmp_path: Path) -> None:
        records.append(workspace=tmp_path, table="items", rows=(row("a"), row("b")))
        assert records.read(workspace=tmp_path, table="items")["stored"] == 2

    def test_it_survives_the_call_that_wrote_it(self, tmp_path: Path) -> None:
        """The whole point: the response is no longer the transport."""
        records.append(workspace=tmp_path, table="items", rows=(row("a"),))
        records.append(workspace=tmp_path, table="items", rows=(row("b"),))
        stored = records.read(workspace=tmp_path, table="items")["records"]
        assert {item["item_id"] for item in stored} == {"a", "b"}

    def test_open_ended_maps_come_back_as_maps(self, tmp_path: Path) -> None:
        """A table wants columns; `values` keys come from the naming convention."""
        records.append(workspace=tmp_path, table="items", rows=(row("a", vendor="Globex"),))
        stored = records.read(workspace=tmp_path, table="items")["records"][0]
        assert stored["values"] == {"vendor": "Globex"}
        assert stored["confidences"] == {"vendor": 0.9}

    def test_reading_a_document_twice_leaves_one_record(self, tmp_path: Path) -> None:
        records.append(workspace=tmp_path, table="items", rows=(row("a", vendor="ACME"),))
        result = records.append(
            workspace=tmp_path, table="items", rows=(row("a", vendor="Corrected"),)
        )
        assert result["replaced"] == 1 and result["stored"] == 1
        stored = records.read(workspace=tmp_path, table="items")["records"][0]
        assert stored["values"]["vendor"] == "Corrected", "and the correction wins"

    def test_an_empty_table_reads_as_empty_rather_than_missing(self, tmp_path: Path) -> None:
        assert records.read(workspace=tmp_path, table="items") == {
            "table": "items",
            "records": [],
            "stored": 0,
        }

    def test_a_row_without_its_key_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="item_id"):
            records.append(workspace=tmp_path, table="items", rows=({"vendor": "ACME"},))

    def test_rows_come_back_in_a_stable_order(self, tmp_path: Path) -> None:
        """A plan built from the table must not depend on the order it was written."""
        records.append(workspace=tmp_path, table="items", rows=(row("c"), row("a")))
        records.append(workspace=tmp_path, table="items", rows=(row("b"),))
        stored = records.read(workspace=tmp_path, table="items")["records"]
        assert [item["item_id"] for item in stored] == ["a", "b", "c"]


class TestContainment:
    """It writes, so where it may write is the whole question."""

    @pytest.mark.parametrize("name", ["../escape", "a/b", "a\\b", ".hidden", ""])
    def test_a_table_name_that_is_a_path_is_refused(self, tmp_path: Path, name: str) -> None:
        with pytest.raises(ValueError, match="table name"):
            records.append(workspace=tmp_path, table=name, rows=(row("a"),))

    def test_everything_lands_under_the_workspace(self, tmp_path: Path) -> None:
        workspace = tmp_path / "run" / "work"
        records.append(workspace=workspace, table="items", rows=(row("a"),))
        written = list(tmp_path.rglob("*.parquet"))
        assert written and all(workspace in path.parents for path in written)

    def test_the_family_is_the_only_writer_outside_filesystem_mutation(self) -> None:
        from shakespeare.contracts import OperatorFamily
        from shakespeare.registry import WRITING_FAMILIES

        assert WRITING_FAMILIES == {
            OperatorFamily.FILESYSTEM_MUTATION,
            OperatorFamily.RECORD_STORE,
        }

    def test_a_writing_family_is_never_auto_admissible(self) -> None:
        """A component that writes is a human's decision, however low its computed risk."""
        from shakespeare.contracts import AUTO_ADMISSIBLE_FAMILIES
        from shakespeare.registry import WRITING_FAMILIES

        assert not (AUTO_ADMISSIBLE_FAMILIES & WRITING_FAMILIES)


class TestThroughTheExecutor:
    def _run(self, tmp_path: Path, *invocations: Invocation):
        operators = build_registry()
        return Executor(operators, Verifier(operators)).execute(
            Composition(domain_id="transcribe", rationale="store", invocations=invocations),
            DomainSpec(
                id="transcribe",
                scope="read into records",
                catalog=frozenset({"record.append", "record.read"}),
            ),
            stage_inputs={},
            config={},
            workspace=tmp_path / "work",
            budget=Budget(envelope=BudgetEnvelope(operator_calls="20"), items=0),
        )

    def test_a_capability_can_append_and_read_in_one_composition(self, tmp_path: Path) -> None:
        results = self._run(
            tmp_path,
            Invocation(
                invocation_id="w",
                operator="record.append",
                parameters={"rows": [row("a"), row("b")]},
            ),
            Invocation(invocation_id="r", operator="record.read", parameters={}),
        )
        assert all(item.succeeded for item in results)
        assert results[1].output is not None
        assert results[1].output["stored"] == 2

    def test_the_store_is_the_run_workspace_and_not_a_supplied_path(
        self, tmp_path: Path
    ) -> None:
        """The path is the runtime's, so no argument can move it."""
        results = self._run(
            tmp_path,
            Invocation(
                invocation_id="w",
                operator="record.append",
                parameters={"rows": [row("a")], "table": "items"},
            ),
        )
        assert results[0].output is not None
        assert str(tmp_path / "work") in results[0].output["path"]
