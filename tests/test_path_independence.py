"""The guard against re-scripting the agents.

The design's central correction was that a stage constrains the *type* of work, never the
method. This is the test that keeps it honest: two agents that reach the same field values
by deliberately different operator sequences must produce byte-identical plans.

If this test ever needs the two agents to agree on a route, the design has drifted back
toward scripting and the plan has stopped being a property of the operators.
"""

from __future__ import annotations

from pathlib import Path

from shakespeare.agent import FakeDomainAgent
from shakespeare.contracts import Composition, Invocation

from test_rename_files import INVOICES, SPEC, _values, build, build_agents, seed_invoices


def _direct(items: list[dict[str, object]]) -> dict[str, FakeDomainAgent]:
    """Route A: read every file with the fallback chain, render in one step."""
    agents = build_agents(items)
    agents["content_acquisition"] = FakeDomainAgent().queue(
        "content_acquisition",
        Composition(
            domain_id="content_acquisition",
            invocations=(
                Invocation(
                    invocation_id="extract",
                    operator="doc.extract",
                    selections={"extract": "auto_chain"},
                    inputs=("root", "items"),
                ),
            ),
        ),
    )
    return agents


def _staged(items: list[dict[str, object]]) -> dict[str, FakeDomainAgent]:
    """Route B: pin the text backend, normalise separately, render from a frozen spec.

    Three invocations where route A used one, a different Hydra selection, and an extra
    operator — but the same values reach the renderer.
    """
    agents = build_agents(items)
    agents["content_acquisition"] = FakeDomainAgent().queue(
        "content_acquisition",
        Composition(
            domain_id="content_acquisition",
            invocations=(
                Invocation(
                    invocation_id="extract",
                    operator="doc.extract",
                    selections={"extract": "pdf_text"},
                    inputs=("root", "items"),
                ),
                Invocation(
                    invocation_id="tidy",
                    operator="text.normalize",
                    parameters={"values": {"probe": "  spacing   noise "}},
                ),
            ),
        ),
    )
    agents["field_resolution"] = FakeDomainAgent().queue(
        "field_resolution",
        Composition(
            domain_id="field_resolution",
            invocations=(
                Invocation(
                    invocation_id="prenormalise",
                    operator="text.normalize",
                    parameters={"values": {"vendor": "ACME Corporation"}},
                ),
                Invocation(
                    invocation_id="render",
                    operator="name.render",
                    inputs=("spec",),
                    parameters={"items": items, "spec": SPEC},
                ),
            ),
        ),
    )
    return agents


def _plan_via(tmp_path: Path, route) -> tuple[list[tuple[str, str, str | None]], int]:
    source = seed_invoices(tmp_path / "in", INVOICES)
    agents = route(_values(source, INVOICES))
    runtime, request, audit, _ = build(tmp_path, agents=agents)
    result = runtime.run(request, commit=False)
    assert result.plan is not None, result.detail
    decisions = sorted(
        (entry.source_ref, str(entry.action), getattr(entry, "target_relpath", None))
        for entry in result.plan.entries
    )
    invocations = sum(
        len(items) for outcome in result.stages for items in outcome.results.values()
    )
    audit.close()
    return decisions, invocations


class TestPathIndependence:
    def test_different_routes_produce_identical_plans(self, tmp_path: Path) -> None:
        direct, direct_calls = _plan_via(tmp_path / "a", _direct)
        staged, staged_calls = _plan_via(tmp_path / "b", _staged)

        assert staged_calls > direct_calls, (
            "the two routes must genuinely differ, or this proves nothing"
        )
        assert direct == staged, (
            "the same field values reached the renderer by different routes, so the plan "
            "must be identical: naming is a property of the operators, not of the route"
        )

    def test_both_routes_produce_the_expected_names(self, tmp_path: Path) -> None:
        """Pinned rather than merely equal, so a shared regression cannot pass silently."""
        expected = [
            ("2024/q1/scan001.pdf", "changed",
             "2024/q1/202401, ACME Corporation, INV-99812, PO-44117.pdf"),
            ("2024/q1/scan002.pdf", "changed",
             "2024/q1/202401, ACME Corporation, INV-99813, PO-44118.pdf"),
            ("2024/q2/scan003.pdf", "changed",
             "2024/q2/202404, Globex Ltd, INV-20001, PO-77310.pdf"),
        ]
        assert _plan_via(tmp_path / "a", _direct)[0] == expected
        assert _plan_via(tmp_path / "b", _staged)[0] == expected

    def test_neither_route_is_privileged_by_the_runtime(self) -> None:
        """The driver must not contain a preferred sequence for any domain."""
        import shakespeare

        root = Path(shakespeare.__file__).parent
        for module in ("runtime.py", "executor.py", "verifier.py"):
            source = (root / module).read_text()
            for operator in ("doc.extract", "name.render", "plan.assemble", "auto_chain"):
                assert operator not in source, f"{module} names {operator}"
