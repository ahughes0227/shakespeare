"""Authorization between decision and action."""

from __future__ import annotations

import pytest
from system.components.builtin import build_registry
from system.contracts import (
    Composition,
    DomainGoal,
    DomainSpec,
    ErrorCode,
    Invocation,
    Obligation,
    SkipDecision,
    StagePlan,
    StageSpec,
)
from system.runtime.verifier import Denial, Verifier, unmet

ACQUISITION = DomainSpec(
    id="content_acquisition",
    scope="Obtain the best available text for every file.",
    skippable=False,
    catalog=frozenset({"doc.extract", "text.normalize"}),
    config_groups=frozenset({"extract", "confidence"}),
)
ENRICHMENT = DomainSpec(
    id="metadata_enrichment",
    scope="Read embedded metadata where cheaply available.",
    skippable=True,
    catalog=frozenset({"doc.extract"}),
    config_groups=frozenset({"extract"}),
)
STAGE = StageSpec(
    name="extract",
    version="1.0.0",
    purpose="Acquire text.",
    goal="Every file has text or a reason.",
    input_contract="FileInventory",
    output_contract="ExtractedContent",
    domains=(ACQUISITION, ENRICHMENT),
    obligations=("every_item_has_text_or_reason",),
)


@pytest.fixture
def verifier() -> Verifier:
    return Verifier(build_registry())


def _goal(domain_id: str) -> DomainGoal:
    return DomainGoal(domain_id=domain_id, goal="do the work", success_criterion="all items")


class TestStagePlanAuthorization:
    def test_accepts_a_plan_accounting_for_every_domain(self, verifier: Verifier) -> None:
        verifier.verify_stage_plan(
            StagePlan(
                activated=(_goal("content_acquisition"),),
                skipped=(SkipDecision(domain_id="metadata_enrichment", reason="not needed"),),
            ),
            STAGE,
        )

    def test_refuses_to_skip_a_non_skippable_domain(self, verifier: Verifier) -> None:
        """A safety or gating domain must not be plannable away."""
        with pytest.raises(Denial, match="not skippable"):
            verifier.verify_stage_plan(
                StagePlan(
                    activated=(_goal("metadata_enrichment"),),
                    skipped=(SkipDecision(domain_id="content_acquisition", reason="slow"),),
                ),
                STAGE,
            )

    def test_refuses_a_silently_unaccounted_domain(self, verifier: Verifier) -> None:
        with pytest.raises(Denial, match="activated or skipped"):
            verifier.verify_stage_plan(StagePlan(activated=(_goal("content_acquisition"),)), STAGE)

    def test_refuses_an_invented_domain(self, verifier: Verifier) -> None:
        with pytest.raises(Denial, match="not in extract"):
            verifier.verify_stage_plan(StagePlan(activated=(_goal("invented"),)), STAGE)


class TestRerunProgress:
    def test_identical_rerun_is_refused(self, verifier: Verifier) -> None:
        plan = StagePlan(activated=(_goal("content_acquisition"),))
        with pytest.raises(Denial, match="would not progress"):
            verifier.verify_rerun(plan, plan)

    def test_revised_rerun_is_allowed(self, verifier: Verifier) -> None:
        first = StagePlan(activated=(_goal("content_acquisition"),))
        second = StagePlan(
            activated=(
                DomainGoal(
                    domain_id="content_acquisition",
                    goal="retry the 40 items that returned ocr_unavailable",
                    success_criterion="all items",
                ),
            )
        )
        verifier.verify_rerun(first, second)


class TestCompositionAuthorization:
    def _composition(self, **invocation: object) -> Composition:
        return Composition(
            domain_id="content_acquisition",
            invocations=(Invocation(invocation_id="a", **invocation),),  # type: ignore[arg-type]
        )

    def test_accepts_a_catalogued_operator(self, verifier: Verifier) -> None:
        verifier.verify_composition(
            self._composition(operator="doc.extract", selections={"extract": "pdf_text"}),
            ACQUISITION,
        )

    def test_refuses_an_operator_outside_the_catalog(self, verifier: Verifier) -> None:
        with pytest.raises(Denial, match="outside the catalog"):
            verifier.verify_composition(self._composition(operator="fs.scan"), ACQUISITION)

    def test_refuses_a_mutation_operator_even_if_a_catalog_lists_it(
        self, verifier: Verifier
    ) -> None:
        """Belt and braces.

        A stage package should never grant fs.commit, but if one did, the runtime-only
        guard must still stop an agent writing before Review has run.
        """
        misconfigured = ACQUISITION.model_copy(
            update={"catalog": frozenset({"doc.extract", "fs.commit"})}
        )
        with pytest.raises(Denial, match="reserved to the runtime"):
            verifier.verify_composition(self._composition(operator="fs.commit"), misconfigured)

    def test_refuses_an_unregistered_operator(self, verifier: Verifier) -> None:
        rogue = ACQUISITION.model_copy(update={"catalog": frozenset({"doc.invented"})})
        with pytest.raises(Denial, match="unknown operator"):
            verifier.verify_composition(self._composition(operator="doc.invented"), rogue)

    def test_refuses_a_config_group_the_domain_was_not_granted(self, verifier: Verifier) -> None:
        with pytest.raises(Denial, match="not granted"):
            verifier.verify_composition(
                self._composition(operator="doc.extract", selections={"collision": "fail"}),
                ACQUISITION,
            )

    def test_refuses_hydra_injection_in_parameters(self, verifier: Verifier) -> None:
        with pytest.raises(Denial, match="unsafe parameter"):
            verifier.verify_composition(
                self._composition(operator="doc.extract", parameters={"_target_": "os.system"}),
                ACQUISITION,
            )

    def test_refuses_a_composition_issued_to_another_domain(self, verifier: Verifier) -> None:
        with pytest.raises(Denial, match="issued to"):
            verifier.verify_composition(self._composition(operator="doc.extract"), ENRICHMENT)

    def test_refuses_more_calls_than_the_budget_allows(self, verifier: Verifier) -> None:
        composition = Composition(
            domain_id="content_acquisition",
            invocations=tuple(
                Invocation(invocation_id=f"i{index}", operator="doc.extract")
                for index in range(5)
            ),
        )
        with pytest.raises(Denial) as caught:
            verifier.verify_composition(composition, ACQUISITION, operator_call_budget=3)
        assert caught.value.code is ErrorCode.BUDGET_EXHAUSTED

    def test_goal_text_cannot_widen_the_surface(self, verifier: Verifier) -> None:
        """A DomainGoal has no catalog field, so a persuasive goal changes nothing.

        This asserts the structural property rather than the behaviour: if a catalog or
        config field were ever added to DomainGoal, goals would become an authority
        escalation path.
        """
        forbidden = {"catalog", "config_groups", "allowed_operators", "budget"}
        assert not forbidden & set(DomainGoal.model_fields)


class TestObligations:
    def test_missing_evidence_fails_closed(self, verifier: Verifier) -> None:
        """An unevaluated obligation is not a satisfied one."""
        results = verifier.check_obligations(
            (Obligation(id="balanced", description="d", checker="balanced"),), {}
        )
        assert unmet(results) == ("balanced",)
        assert "no evidence" in str(results[0].detail)

    def test_passing_evidence_satisfies(self, verifier: Verifier) -> None:
        results = verifier.check_obligations(
            (Obligation(id="balanced", description="d", checker="balanced"),),
            {"balanced": {"entries": [{"item_id": "1"}], "scanned": 1}},
        )
        assert unmet(results) == ()
