from __future__ import annotations

import pytest
from pydantic import ValidationError
from shakespeare.contracts import (
    Allowance,
    BudgetEnvelope,
    ChangeAction,
    ChangeEntry,
    ChangePlan,
    Composition,
    DomainGoal,
    Invocation,
    SkipDecision,
    StagePlan,
    TelemetryEnvelope,
    WorkflowSpec,
    content_digest,
)


class TestAllowance:
    def test_parses_base_only(self) -> None:
        assert Allowance.parse("12").resolve(100) == 12

    def test_parses_per_item(self) -> None:
        assert Allowance.parse("20 + 6*n").resolve(10) == 80

    def test_rejects_arbitrary_expression(self) -> None:
        with pytest.raises(ValueError):
            Allowance.parse("20 + 6*n**2")

    def test_budget_coerces_strings(self) -> None:
        budget = BudgetEnvelope(operator_calls="4 + 1*n")
        assert budget.operator_calls.resolve(3) == 7


class TestComposition:
    def test_edges_follow_declared_inputs(self) -> None:
        composition = Composition(
            domain_id="d",
            invocations=(
                Invocation(invocation_id="a", operator="fs.scan"),
                Invocation(invocation_id="b", operator="doc.extract", inputs=("a",)),
                Invocation(invocation_id="c", operator="text.normalize", inputs=("b",)),
            ),
        )
        assert composition.edges() == (("a", "b"), ("b", "c"))

    def test_rejects_forward_reference(self) -> None:
        with pytest.raises(ValidationError):
            Composition(
                domain_id="d",
                invocations=(
                    Invocation(invocation_id="a", operator="fs.scan", inputs=("b",)),
                    Invocation(invocation_id="b", operator="doc.extract"),
                ),
            )

    def test_rejects_duplicate_invocation_id(self) -> None:
        with pytest.raises(ValidationError):
            Composition(
                domain_id="d",
                invocations=(
                    Invocation(invocation_id="a", operator="fs.scan"),
                    Invocation(invocation_id="a", operator="fs.scan"),
                ),
            )

    def test_stage_input_reference_is_not_an_edge(self) -> None:
        composition = Composition(
            domain_id="d",
            invocations=(Invocation(invocation_id="a", operator="fs.scan", inputs=("inventory",)),),
        )
        assert composition.edges() == ()


class TestStagePlan:
    def test_rejects_domain_both_activated_and_skipped(self) -> None:
        with pytest.raises(ValidationError):
            StagePlan(
                activated=(DomainGoal(domain_id="x", goal="g", success_criterion="c"),),
                skipped=(SkipDecision(domain_id="x", reason="r"),),
            )


class TestWorkflowSpec:
    def test_accepts_pinned_spine(self) -> None:
        spec = WorkflowSpec(
            id="rename_files",
            version="1.0.0",
            spine=("intake@1.0.0", "review@1.0.0"),
            commit_after="review",
        )
        assert spec.spine[0] == "intake@1.0.0"

    def test_rejects_unpinned_stage(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowSpec(id="w", version="1.0.0", spine=("intake",), commit_after="intake")

    def test_rejects_commit_after_outside_spine(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowSpec(
                id="w", version="1.0.0", spine=("intake@1.0.0",), commit_after="review"
            )

    def test_rejects_repeated_stage(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowSpec(
                id="w",
                version="1.0.0",
                spine=("intake@1.0.0", "intake@1.1.0"),
                commit_after="intake",
            )


class TestChangePlan:
    def _plan(self, *entries: ChangeEntry) -> ChangePlan:
        return ChangePlan(
            run_id="r",
            workflow_id="w",
            workflow_digest="d",
            decision_digest="s",
            entries=entries,
        )

    def test_balanced_requires_one_entry_per_item(self) -> None:
        plan = self._plan(
            ChangeEntry(item_id="1", source_ref="a", action=ChangeAction.CHANGED),
            ChangeEntry(item_id="2", source_ref="b", action=ChangeAction.UNRESOLVED),
        )
        assert plan.balanced(2)
        assert not plan.balanced(3)

    def test_duplicate_item_is_not_balanced(self) -> None:
        plan = self._plan(
            ChangeEntry(item_id="1", source_ref="a", action=ChangeAction.CHANGED),
            ChangeEntry(item_id="1", source_ref="b", action=ChangeAction.CHANGED),
        )
        assert not plan.balanced(2)


class TestTelemetryEnvelope:
    def test_rejects_non_digest_in_digests(self) -> None:
        with pytest.raises(ValidationError):
            TelemetryEnvelope(run_id="r", span="s", digests={"content": "ACME Corporation"})

    def test_accepts_real_digest(self) -> None:
        envelope = TelemetryEnvelope(
            run_id="r", span="s", digests={"content": content_digest("ACME Corporation")}
        )
        assert "ACME" not in envelope.model_dump_json()


class TestBindingRefusals:
    """A binding names a reference. A literal is refused, not reclassified.

    Reclassifying it as a parameter would let a binding smuggle in a path the run was
    never given, and what may be read is a trust question rather than a shape question.
    """

    def test_a_literal_path_is_refused_with_advice(self) -> None:
        with pytest.raises(ValidationError) as caught:
            Invocation(invocation_id="a", operator="fs.scan", bindings={"root": "/etc/passwd"})
        message = str(caught.value)
        assert "literal value, not a reference" in message
        assert "put it in parameters instead" in message

    def test_a_dotted_reference_is_accepted(self) -> None:
        invocation = Invocation(
            invocation_id="b", operator="doc.extract", bindings={"items": "step1.window"}
        )
        assert invocation.bindings["items"] == "step1.window"

    def test_a_hyphenated_invocation_id_is_a_valid_label(self) -> None:
        """Invocation ids are labels, not Python identifiers."""
        invocation = Invocation(
            invocation_id="c", operator="doc.extract", bindings={"items": "inv-1.window"}
        )
        assert invocation.bindings["items"] == "inv-1.window"

    def test_an_empty_inputs_mapping_becomes_no_inputs(self) -> None:
        assert Invocation(invocation_id="d", operator="fs.scan", inputs={}).inputs == ()
