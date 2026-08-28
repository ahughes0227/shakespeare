"""The guard against re-scripting the capabilities.

A capability owns its own decomposition, so two capabilities that reach the same field
values by deliberately different internal routes must produce identical plans. If this
test ever needs them to agree on a route, adaptive organization has stopped being the
capability's own.
"""

from __future__ import annotations

from pathlib import Path

from system.contracts import Invocation

from harness import SPEC, build, org, rename_agent, seed_invoices, values_for


def _direct(items: list[dict[str, object]]):
    """Route A: read everything with the fallback chain, render in one round."""
    return rename_agent(items)


def _staged(items: list[dict[str, object]]):
    """Route B: pin the backend, normalise separately, render over two rounds.

    Four component calls where route A used two, a different Hydra selection, and an
    extra round — but the same values reach the renderer.
    """
    agent = rename_agent(items)
    agent.plans["acquire"] = [
        org(
            Invocation(
                invocation_id="extract",
                operator="doc.extract",
                selections={"extract": "pdf_text"},
                inputs=("root", "items"),
            ),
            intent="try the text layer first",
            sufficient=False,
            publishes=None,
        ),
        org(
            Invocation(
                invocation_id="tidy",
                operator="text.normalize",
                parameters={"values": {"probe": "  spacing   noise "}},
            ),
            intent="tidy what came back",
            publishes="ExtractedContent",
        ),
    ]
    agent.plans["resolve"] = [
        org(
            Invocation(
                invocation_id="prenormalise",
                operator="text.normalize",
                parameters={"values": {"vendor": "ACME Corporation"}},
            ),
            intent="normalise first",
            sufficient=False,
            publishes=None,
        ),
        org(
            Invocation(
                invocation_id="render",
                operator="name.render",
                inputs=("spec",),
                parameters={"items": items, "spec": SPEC},
            ),
            intent="then render",
            publishes="ResolvedNames",
        ),
    ]
    return agent


def _plan_via(tmp_path: Path, route) -> tuple[list[tuple[str, str, str | None]], int]:
    source = seed_invoices(tmp_path / "in")
    agent = route(values_for(source))
    runtime, request, audit, _ = build(tmp_path, agents={"*": agent})
    result = runtime.run(request, commit=False)
    assert result.plan is not None, result.detail
    decisions = sorted(
        (entry.source_ref, str(entry.action), getattr(entry, "target_relpath", None))
        for entry in result.plan.entries
    )
    rounds = sum(len(attempt.outcome.rounds) for attempt in result.attempts)
    audit.close()
    return decisions, rounds


class TestPathIndependence:
    def test_different_routes_produce_identical_plans(self, tmp_path: Path) -> None:
        direct, direct_rounds = _plan_via(tmp_path / "a", _direct)
        staged, staged_rounds = _plan_via(tmp_path / "b", _staged)

        assert staged_rounds > direct_rounds, (
            "the two routes must genuinely differ, or this proves nothing"
        )
        assert direct == staged, (
            "the same values reached the renderer by different routes, so the plan must "
            "be identical: naming is a property of the components, not of the route"
        )

    def test_both_routes_produce_the_expected_names(self, tmp_path: Path) -> None:
        expected = [
            (
                "2024/q1/scan001.pdf",
                "changed",
                "2024/q1/202401, ACME Corporation, INV-99812, PO-44117.pdf",
            ),
            (
                "2024/q1/scan002.pdf",
                "changed",
                "2024/q1/202401, ACME Corporation, INV-99813, PO-44118.pdf",
            ),
            (
                "2024/q2/scan003.pdf",
                "changed",
                "2024/q2/202404, Globex Ltd, INV-20001, PO-77310.pdf",
            ),
        ]
        assert _plan_via(tmp_path / "a", _direct)[0] == expected
        assert _plan_via(tmp_path / "b", _staged)[0] == expected

    def test_the_loop_privileges_no_route(self) -> None:
        import system

        root = Path(system.__file__).parent
        for module in ("runtime/control.py", "runtime/engine.py", "runtime/gating.py"):
            source = (root / module).read_text()
            for component in ("doc.extract", "name.render", "plan.assemble", "auto_chain"):
                assert component not in source, f"{module} names {component}"
