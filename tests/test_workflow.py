from pathlib import Path

from avathon.data_generation import generate
from avathon.workflow import build_workflow


def _run(data_dir: Path, scenario: str) -> dict:
    generate(data_dir, scenario=scenario)
    app = build_workflow(data_dir)
    config = {"configurable": {"thread_id": f"test-{scenario}"}}
    initial = {
        "trace_id": f"test-{scenario}", "scenario_id": scenario, "sku": "SKU-CRITICAL",
        "site": "Pune-DC", "decision": "pending", "messages": [],
    }
    app.invoke(initial, config=config)
    if scenario == "success":
        app.update_state(config, {"decision": "approve"})
        app.invoke(None, config=config)
    return app.get_state(config).values


def test_verified_delay_requires_human_review_then_allows_simulated_action(tmp_path: Path) -> None:
    state = _run(tmp_path / "success", "success")
    message_types = [message["message_type"] for message in state["messages"]]
    assert "risk_assessment" in message_types
    assert "impact_assessment" in message_types
    assert "approval_request" in message_types
    assert message_types[-1] == "action_proposal"
    assert state["messages"][-1]["payload"]["execution_status"] == "approved_for_simulated_execution"


def test_conflicting_shipment_data_fails_closed_without_recommendation(tmp_path: Path) -> None:
    state = _run(tmp_path / "failure", "conflicting_evidence")
    message_types = [message["message_type"] for message in state["messages"]]
    assert "escalation" in message_types
    assert "approval_request" not in message_types
    assert "action_proposal" not in message_types
