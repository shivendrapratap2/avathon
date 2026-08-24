"""End-to-end graph behaviour: routing, the human gate, and the audit trace."""

from __future__ import annotations

from pathlib import Path

import pytest

from avathon.data_generation import generate
from avathon.llm import reference_planner_factory, rogue_planner_factory
from avathon.workflow import build_workflow


SKU, SITE = "SKU-CRITICAL", "Pune-DC"


def _run(
    tmp_path: Path, scenario: str, *, decision: str = "approve",
    planner_id: str = "planner@example.com", factory=None,
) -> dict:
    data_dir = tmp_path / scenario
    generate(data_dir, scenario=scenario)
    app = build_workflow(
        data_dir, planner_factory=factory or reference_planner_factory(SKU, SITE)
    )
    config = {"configurable": {"thread_id": f"test-{scenario}-{decision}"}}
    app.invoke(
        {
            "trace_id": f"test-{scenario}", "scenario_id": scenario, "sku": SKU,
            "site": SITE, "planner_id": planner_id, "decision": "pending", "messages": [],
        },
        config=config,
    )
    paused = bool(app.get_state(config).next)
    if paused:
        app.update_state(config, {"decision": decision})
        app.invoke(None, config=config)
    state = dict(app.get_state(config).values)
    state["_paused_for_human"] = paused
    return state


def _types(state: dict) -> list[str]:
    return [message["message_type"] for message in state["messages"]]


# -- the working path --------------------------------------------------------


def test_a_verified_delay_pauses_for_a_human_then_records_the_approved_action(
    tmp_path: Path,
) -> None:
    state = _run(tmp_path, "verified_delay")
    types = _types(state)

    assert state["_paused_for_human"], "the graph must interrupt before human review"
    assert "risk_assessment" in types
    assert "impact_assessment" in types
    assert "approval_request" in types
    assert types[-1] == "action_proposal"

    final = state["messages"][-1]["payload"]
    assert final["execution_status"] == "approved_for_simulated_execution"
    assert final["approved_by"] == "planner@example.com"


def test_the_approved_action_cites_the_full_evidence_chain(tmp_path: Path) -> None:
    state = _run(tmp_path, "verified_delay")
    final = state["messages"][-1]
    # One reference each from the risk, impact, and alternate-site queries.
    assert len(final["evidence_refs"]) == 3
    assert all(ref.startswith("duckdb:") for ref in final["evidence_refs"])


def test_no_action_is_ever_recorded_before_the_human_decision(tmp_path: Path) -> None:
    state = _run(tmp_path, "verified_delay")
    types = _types(state)
    assert types.index("human_decision") < types.index("action_proposal")


def test_the_proposal_is_marked_blocked_while_it_awaits_a_human(tmp_path: Path) -> None:
    state = _run(tmp_path, "verified_delay")
    approval = next(
        m for m in state["messages"] if m["message_type"] == "approval_request"
    )
    assert approval["payload"]["execution_status"] == "blocked_pending_human_approval"


def test_ample_cover_proposes_monitoring_rather_than_moving_stock(tmp_path: Path) -> None:
    state = _run(tmp_path, "healthy_stock")
    assert state["proposal"]["action"] == "monitor"
    assert state["proposal"]["quantity"] == 0


def test_a_transfer_never_exceeds_the_surplus_it_cites(tmp_path: Path) -> None:
    proposal = _run(tmp_path, "verified_delay")["proposal"]
    assert proposal["quantity"] <= proposal["alternate_site_surplus_units"]
    assert proposal["quantity"] <= proposal["policy_cap_units"]
    assert proposal["source_site"] == "Mumbai-DC"


# -- fail closed -------------------------------------------------------------


def test_contradictory_updates_fail_closed_without_a_recommendation(tmp_path: Path) -> None:
    state = _run(tmp_path, "conflicting_evidence")
    types = _types(state)
    assert state["escalation_type"] == "conflicting_evidence"
    assert "approval_request" not in types
    assert "action_proposal" not in types
    assert not state["_paused_for_human"]


def test_absent_evidence_fails_closed(tmp_path: Path) -> None:
    state = _run(tmp_path, "no_evidence")
    assert state["escalation_type"] == "no_evidence"
    assert "action_proposal" not in _types(state)


def test_an_out_of_scope_planner_is_blocked_before_any_query_runs(tmp_path: Path) -> None:
    state = _run(tmp_path, "verified_delay", factory=rogue_planner_factory(SKU, SITE))
    types = _types(state)
    assert state["escalation_type"] == "guardrail_violation"
    assert "guardrail_block" in types
    assert "action_proposal" not in types

    block = next(m for m in state["messages"] if m["message_type"] == "guardrail_block")
    assert block["payload"]["executed"] is False
    assert block["payload"]["violation"] == "out_of_scope_sku"


# -- regressions: complicated is not the same as untrustworthy ---------------


@pytest.mark.parametrize("scenario", ["multi_po_clean", "superseded_update"])
def test_normal_operating_complexity_does_not_escalate(tmp_path: Path, scenario: str) -> None:
    """Several open POs, and a corrected update, are routine — not contradictions."""
    state = _run(tmp_path, scenario)
    assert state.get("escalation_type") is None
    assert "approval_request" in _types(state)


def test_several_open_pos_are_all_counted_in_the_risk_assessment(tmp_path: Path) -> None:
    state = _run(tmp_path, "multi_po_clean")
    assert state["risk"]["open_po_count"] == 2
    assert state["risk"]["delayed_po_count"] == 1


# -- the human gate is a real gate -------------------------------------------


def test_a_rejected_proposal_is_distinguishable_from_a_data_failure(tmp_path: Path) -> None:
    state = _run(tmp_path, "verified_delay", decision="reject")
    assert state["escalation_type"] == "human_rejected"
    assert "action_proposal" not in _types(state)

    escalation = state["messages"][-1]["payload"]
    assert escalation["action_executed"] is False
    assert escalation["escalation_type"] == "human_rejected"


def test_the_approval_is_attributed_to_a_named_planner(tmp_path: Path) -> None:
    state = _run(tmp_path, "verified_delay", planner_id="asha.k@example.com")
    decision = next(
        m for m in state["messages"] if m["message_type"] == "human_decision"
    )
    assert decision["payload"]["planner_id"] == "asha.k@example.com"


def test_escalation_types_are_machine_readable_not_prose(tmp_path: Path) -> None:
    for scenario, expected in [
        ("conflicting_evidence", "conflicting_evidence"),
        ("no_evidence", "no_evidence"),
    ]:
        state = _run(tmp_path, scenario)
        assert state["messages"][-1]["payload"]["escalation_type"] == expected


# -- trace integrity ---------------------------------------------------------


def test_tool_failures_are_attributed_to_the_executor_not_the_model(tmp_path: Path) -> None:
    """A DuckDB error must never be labelled as an OpenAI failure in the audit log."""
    data_dir = tmp_path / "broken"
    generate(data_dir, scenario="verified_delay")
    (data_dir / "inventory.csv").write_text("not,a,valid\nschema\n", encoding="utf-8")

    app = build_workflow(data_dir, planner_factory=reference_planner_factory(SKU, SITE))
    config = {"configurable": {"thread_id": "test-broken"}}
    app.invoke(
        {
            "trace_id": "test-broken", "scenario_id": "broken", "sku": SKU, "site": SITE,
            "planner_id": "planner@example.com", "decision": "pending", "messages": [],
        },
        config=config,
    )
    state = app.get_state(config).values

    assert state["escalation_type"] == "tool_failure"
    failure = next(
        m for m in state["messages"] if m["payload"].get("status") == "failed"
    )
    assert failure["payload"]["executed_by"] == "local_duckdb"


def test_every_evidence_backed_message_carries_a_query_id(tmp_path: Path) -> None:
    state = _run(tmp_path, "verified_delay")
    for message in state["messages"]:
        if message["message_type"] in ("risk_assessment", "impact_assessment"):
            assert message["evidence_refs"]


def test_tool_results_record_who_selected_and_who_executed(tmp_path: Path) -> None:
    state = _run(tmp_path, "verified_delay")
    results = [
        m for m in state["messages"]
        if m["message_type"] == "tool_result" and "tool_name" in m["payload"]
    ]
    assert results
    for message in results:
        assert message["payload"]["executed_by"] == "local_duckdb"
        assert message["payload"]["selected_by"] == "reference-plan"


def test_the_graph_refuses_to_run_with_no_planner_at_all(tmp_path: Path) -> None:
    data_dir = tmp_path / "noplanner"
    generate(data_dir, scenario="verified_delay")
    with pytest.raises(ValueError, match="api_key"):
        build_workflow(data_dir)
