"""The full graph driven through LLMPlanner, with a simulated model.

The scripted planner exercises the graph but bypasses the model-facing code:
the conversation loop, the narration turn, and the grounding check. These tests
drive the real ``LLMPlanner`` against a stand-in client that behaves the way a
competent model behaves - and, in the last two, the way a careless one does.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from avathon.data_generation import generate
from avathon.llm import LLMPlanner
from avathon.workflow import build_workflow


SKU, SITE = "SKU-CRITICAL", "Pune-DC"

#: What a competent planner calls, keyed by a phrase unique to each objective.
#: Every objective mentions the delay, so the marker has to be more specific
#: than that - matching loosely is how a stub silently tests the wrong thing.
EXPECTED_TOOL = {
    "Establish two things": ("supplier_exposure", {"sku": SKU, "site": SITE}),
    "Retrieve recent daily demand": (
        "demand_history", {"sku": SKU, "site": SITE, "lookback_days": 21},
    ),
    "stock other sites hold": (
        "alternate_site_availability", {"sku": SKU, "exclude_site": SITE},
    ),
}


def _function_call(name: str, arguments: dict, call_id: str) -> SimpleNamespace:
    payload = json.dumps(arguments)
    return SimpleNamespace(
        type="function_call", name=name, arguments=payload, call_id=call_id,
        model_dump=lambda **_: {
            "type": "function_call", "name": name, "arguments": payload,
            "call_id": call_id,
        },
    )


class SimulatedModel:
    """Reads the objective, picks the matching tool, then stops. Narrates on request."""

    def __init__(self, narrative: str = ""):
        self.narrative = narrative
        self.requests: list[dict] = []
        self._counter = 0

    @property
    def responses(self):
        return self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        self._counter += 1

        if not kwargs.get("tools"):  # the narration turn
            return SimpleNamespace(id=f"nar-{self._counter}", output=[],
                                   output_text=self.narrative)

        objective = kwargs["input"][0]["content"]
        already_called = any(
            isinstance(item, dict) and item.get("type") == "function_call"
            for item in kwargs["input"]
        )
        if already_called:
            return SimpleNamespace(id=f"stop-{self._counter}", output=[],
                                   output_text="Evidence retrieved.")

        published = {tool["name"] for tool in kwargs["tools"]}
        for hint, (name, arguments) in EXPECTED_TOOL.items():
            if hint in objective and name in published:
                return SimpleNamespace(
                    id=f"call-{self._counter}", output=[
                        _function_call(name, arguments, f"c{self._counter}")
                    ],
                )
        return SimpleNamespace(id=f"idle-{self._counter}", output=[], output_text="")


def _run(tmp_path: Path, scenario: str, model: SimulatedModel, decision: str = "approve"):
    data_dir = tmp_path / scenario
    generate(data_dir, scenario=scenario)
    planner = LLMPlanner(api_key="unused", model="gpt-4o-mini", client=model)
    app = build_workflow(data_dir, planner=planner)
    config = {"configurable": {"thread_id": f"live-{scenario}-{decision}"}}
    app.invoke(
        {
            "trace_id": f"live-{scenario}", "scenario_id": scenario, "sku": SKU,
            "site": SITE, "planner_id": "planner@example.com",
            "decision": "pending", "messages": [],
        },
        config=config,
    )
    if app.get_state(config).next:
        app.update_state(config, {"decision": decision})
        app.invoke(None, config=config)
    return app.get_state(config).values


def test_a_model_driven_run_reaches_an_approved_action(tmp_path: Path) -> None:
    state = _run(tmp_path, "verified_delay", SimulatedModel("A delay was confirmed."))
    types = [message["message_type"] for message in state["messages"]]

    assert types[-1] == "action_proposal"
    assert state["proposal"]["action"] == "transfer_stock"
    assert types.count("planner_step") == 3, "each agent plans independently"


def test_the_lookback_the_model_chose_is_the_one_that_was_used(tmp_path: Path) -> None:
    state = _run(tmp_path, "verified_delay", SimulatedModel())
    assert state["impact"]["lookback_days"] == 21


def test_the_trace_records_the_model_that_selected_each_tool(tmp_path: Path) -> None:
    state = _run(tmp_path, "verified_delay", SimulatedModel())
    results = [
        message for message in state["messages"]
        if message["message_type"] == "tool_result" and "tool_name" in message["payload"]
    ]
    assert len(results) == 3
    for message in results:
        assert message["payload"]["selected_by"] == "gpt-4o-mini"
        assert message["payload"]["executed_by"] == "local_duckdb"


def test_response_ids_are_captured_for_every_agent(tmp_path: Path) -> None:
    state = _run(tmp_path, "verified_delay", SimulatedModel())
    for message in state["messages"]:
        if message["message_type"] == "planner_step":
            assert message["payload"]["response_ids"]


def test_a_grounded_explanation_reaches_the_planner(tmp_path: Path) -> None:
    # 5 (delay) and 500 (safety stock) are fixed by the scenario, not by the seed.
    narrative = "A verified 5 day delay leaves the site short of its 500 unit floor."
    state = _run(tmp_path, "verified_delay", SimulatedModel(narrative))
    impact = next(
        m for m in state["messages"] if m["message_type"] == "impact_assessment"
    )
    assert impact["narrative"] == narrative
    assert impact["payload"]["narrative_grounding"]["ok"] is True


def test_an_invented_figure_suppresses_the_explanation_but_not_the_decision(
    tmp_path: Path,
) -> None:
    """The decision must survive a hallucinating narrator, because it never read it."""
    model = SimulatedModel("The shortfall is approximately 9412 units.")
    state = _run(tmp_path, "verified_delay", model)

    impact = next(
        m for m in state["messages"] if m["message_type"] == "impact_assessment"
    )
    assert impact["narrative"] == ""
    grounding = impact["payload"]["narrative_grounding"]
    assert grounding["ok"] is False
    assert "9412" in grounding["ungrounded_numbers"]

    # The workflow still completed correctly on the computed figures.
    assert state["messages"][-1]["message_type"] == "action_proposal"
    assert state["proposal"]["quantity"] == 400


def test_a_model_driven_run_still_fails_closed_on_bad_evidence(tmp_path: Path) -> None:
    state = _run(tmp_path, "conflicting_evidence", SimulatedModel())
    assert state["escalation_type"] == "conflicting_evidence"
    assert "action_proposal" not in [m["message_type"] for m in state["messages"]]


def test_a_model_that_refuses_to_gather_evidence_fails_closed(tmp_path: Path) -> None:
    class IdleModel(SimulatedModel):
        def create(self, **kwargs):
            self.requests.append(kwargs)
            return SimpleNamespace(id="idle", output=[], output_text="I'd rather not.")

    state = _run(tmp_path, "verified_delay", IdleModel())
    assert state["escalation_type"] == "missing_required_evidence"
    assert "action_proposal" not in [m["message_type"] for m in state["messages"]]


def test_a_model_transport_failure_escalates_rather_than_guessing(tmp_path: Path) -> None:
    class BrokenModel(SimulatedModel):
        def create(self, **kwargs):
            raise ConnectionError("upstream unavailable")

    state = _run(tmp_path, "verified_delay", BrokenModel())
    assert state["escalation_type"] == "tool_failure"
    assert "action_proposal" not in [m["message_type"] for m in state["messages"]]
