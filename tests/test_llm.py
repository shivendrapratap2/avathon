"""The planner loop and the grounding check on model-written explanations."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from avathon.guardrails import InvestigationScope, ToolCallSafetyError, ToolPolicy
from avathon.llm import LLMPlanner, check_numeric_grounding


class _FakeResponses:
    """Replays scripted Responses API results and records what was sent."""

    def __init__(self, responses: list[SimpleNamespace]):
        self.responses = responses
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _call(name: str, arguments: dict, call_id: str = "call-1") -> SimpleNamespace:
    return SimpleNamespace(
        type="function_call", name=name, arguments=json.dumps(arguments),
        call_id=call_id,
        model_dump=lambda **_: {
            "type": "function_call", "name": name,
            "arguments": json.dumps(arguments), "call_id": call_id,
        },
    )


def _planner(responses: list[SimpleNamespace]) -> LLMPlanner:
    client = SimpleNamespace(responses=_FakeResponses(responses))
    return LLMPlanner(api_key="unused", model="test-model", client=client)


def _policy(**overrides) -> ToolPolicy:
    defaults = dict(
        agent="risk_agent",
        allowed_tools=frozenset({"supplier_exposure", "demand_history"}),
        required_tools=frozenset({"supplier_exposure"}),
        scope=InvestigationScope(sku="SKU-CRITICAL", site="Pune-DC"),
    )
    defaults.update(overrides)
    return ToolPolicy(**defaults)


def _execute(name: str, arguments: dict) -> dict:
    return {"query_id": f"duckdb:{name}", "operation": name, "rows": [arguments]}


# -- the loop ----------------------------------------------------------------


def test_the_planner_runs_tools_until_the_model_stops_asking() -> None:
    planner = _planner([
        SimpleNamespace(id="resp-1", output=[
            _call("supplier_exposure", {"sku": "SKU-CRITICAL", "site": "Pune-DC"})
        ]),
        SimpleNamespace(id="resp-2", output=[
            _call("demand_history",
                  {"sku": "SKU-CRITICAL", "site": "Pune-DC", "lookback_days": 14}, "call-2")
        ]),
        SimpleNamespace(id="resp-3", output=[], output_text="A five-day delay is confirmed."),
    ])

    run = planner.gather(objective="investigate", policy=_policy(), execute=_execute)

    assert run.called_tools == {"supplier_exposure", "demand_history"}
    assert run.tool_call_count == 2
    assert run.response_ids == ["resp-1", "resp-2", "resp-3"]
    assert run.closing_statement == "A five-day delay is confirmed."


def test_the_model_chooses_its_own_lookback_within_bounds() -> None:
    planner = _planner([
        SimpleNamespace(id="r1", output=[
            _call("supplier_exposure", {"sku": "SKU-CRITICAL", "site": "Pune-DC"})]),
        SimpleNamespace(id="r2", output=[
            _call("demand_history",
                  {"sku": "SKU-CRITICAL", "site": "Pune-DC", "lookback_days": 56}, "c2")]),
        SimpleNamespace(id="r3", output=[], output_text="done"),
    ])
    run = planner.gather(objective="investigate", policy=_policy(), execute=_execute)
    assert run.evidence["demand_history"]["rows"][0]["lookback_days"] == 56


def test_evidence_is_never_retained_on_the_provider_side() -> None:
    planner = _planner([SimpleNamespace(id="r1", output=[], output_text="nothing to add")])
    policy = _policy(required_tools=frozenset())
    planner.gather(objective="investigate", policy=policy, execute=_execute)
    request = planner.client.responses.calls[0]
    assert request["store"] is False
    assert "previous_response_id" not in request


def test_only_allow_listed_specs_reach_the_model() -> None:
    planner = _planner([SimpleNamespace(id="r1", output=[], output_text="")])
    planner.gather(
        objective="x", policy=_policy(required_tools=frozenset()), execute=_execute
    )
    published = {tool["name"] for tool in planner.client.responses.calls[0]["tools"]}
    assert published == {"supplier_exposure", "demand_history"}


def test_the_step_budget_stops_a_planner_that_will_not_stop() -> None:
    responses = [
        SimpleNamespace(id=f"r{index}", output=[
            _call("demand_history",
                  {"sku": "SKU-CRITICAL", "site": "Pune-DC", "lookback_days": 7 + index},
                  f"c{index}")
        ])
        for index in range(10)
    ]
    planner = _planner(responses)
    policy = _policy(required_tools=frozenset(), max_steps=3)
    run = planner.gather(objective="x", policy=policy, execute=_execute)
    assert run.tool_call_count == 3


# -- the boundary holds inside the loop --------------------------------------


def test_an_out_of_scope_call_is_blocked_before_the_tool_runs() -> None:
    planner = _planner([
        SimpleNamespace(id="r1", output=[
            _call("supplier_exposure", {"sku": "SKU-VOLATILE", "site": "Mumbai-DC"})])
    ])

    def must_not_run(*_args, **_kwargs):
        pytest.fail("The tool ran despite an out-of-scope request.")

    with pytest.raises(ToolCallSafetyError) as error:
        planner.gather(objective="x", policy=_policy(), execute=must_not_run)
    assert error.value.violation == "out_of_scope_sku"


def test_malformed_arguments_are_blocked_before_the_tool_runs() -> None:
    broken = SimpleNamespace(
        type="function_call", name="supplier_exposure", arguments="{not json",
        call_id="c1", model_dump=lambda **_: {"type": "function_call", "call_id": "c1"},
    )
    planner = _planner([SimpleNamespace(id="r1", output=[broken])])

    def must_not_run(*_args, **_kwargs):
        pytest.fail("The tool ran on malformed arguments.")

    with pytest.raises(ToolCallSafetyError) as error:
        planner.gather(objective="x", policy=_policy(), execute=must_not_run)
    assert error.value.violation == "malformed_arguments"


def test_stopping_without_the_required_evidence_fails_closed() -> None:
    planner = _planner([SimpleNamespace(id="r1", output=[], output_text="I'd rather not.")])
    with pytest.raises(ToolCallSafetyError) as error:
        planner.gather(objective="x", policy=_policy(), execute=_execute)
    assert error.value.violation == "missing_required_evidence"


# -- narrative grounding -----------------------------------------------------


FACTS = {
    "sku": "SKU-CRITICAL", "delay_days": 5, "days_of_cover": 1.9,
    "safety_stock_gap_units": 789, "quantity": 400, "revised_eta": "2026-08-26",
}


def test_a_narrative_using_only_supplied_figures_is_grounded() -> None:
    narrative = (
        "The 5-day delay leaves 1.9 days of cover, a shortfall of 789 units against "
        "safety stock. A 400-unit transfer is proposed ahead of the 2026-08-26 ETA."
    )
    assert check_numeric_grounding(narrative, FACTS) == []


def test_an_invented_figure_is_caught() -> None:
    narrative = "Cover is 1.9 days and the shortfall is roughly 1200 units."
    assert check_numeric_grounding(narrative, FACTS) == ["1200"]


def test_a_plausible_but_recomputed_figure_is_caught() -> None:
    """The dangerous case: arithmetic that looks right and is not in the facts."""
    narrative = "At 1.9 days of cover the site runs dry in 46 hours."
    assert check_numeric_grounding(narrative, FACTS) == ["46"]


def test_the_published_tool_schemas_are_valid_for_strict_mode() -> None:
    """Strict mode rejects unlisted required keys, open objects, and JSON-Schema
    validation keywords. A schema that fails here 400s on the first live call."""
    from avathon.tools import TOOL_SPECS

    for spec in TOOL_SPECS:
        parameters = spec["parameters"]
        assert spec["strict"] is True
        assert parameters["additionalProperties"] is False
        assert set(parameters["required"]) == set(parameters["properties"])
        for prop in parameters["properties"].values():
            assert set(prop) <= {"type", "description", "enum"}, prop


def test_the_loop_round_trips_real_openai_sdk_objects() -> None:
    """Guards the conversation format against SDK drift, without a network call."""
    from openai.types.responses import (
        Response, ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText,
    )

    def response(rid: str, output: list) -> Response:
        return Response(
            id=rid, created_at=0, model="gpt-4o-mini", object="response", output=output,
            parallel_tool_calls=False, tool_choice="auto", tools=[], status="completed",
        )

    tool_call = ResponseFunctionToolCall(
        type="function_call", name="supplier_exposure", id="fc_1", call_id="call_abc",
        arguments='{"sku":"SKU-CRITICAL","site":"Pune-DC"}', status="completed",
    )
    message = ResponseOutputMessage(
        id="m1", type="message", role="assistant", status="completed",
        content=[ResponseOutputText(type="output_text", text="Delay confirmed.", annotations=[])],
    )

    planner = _planner([response("r1", [tool_call]), response("r2", [message])])
    run = planner.gather(objective="go", policy=_policy(), execute=_execute)

    assert run.called_tools == {"supplier_exposure"}
    assert run.closing_statement == "Delay confirmed."

    echoed = planner.client.responses.calls[1]["input"]
    assert echoed[1]["type"] == "function_call"
    assert echoed[1]["call_id"] == "call_abc"
    assert echoed[2]["type"] == "function_call_output"
    assert echoed[2]["call_id"] == "call_abc"
    assert "status" not in echoed[1], "a replayed call must not carry a status field"


def test_hex_query_ids_do_not_whitelist_arbitrary_numbers() -> None:
    """A whitelist that grows with every hash is not a check.

    Query IDs are hex digests. Harvesting digit runs out of them would ground
    almost any invented figure.
    """
    facts = {
        "days_of_cover": 1.9,
        "query_id": "duckdb:c6fdb9a27232",
        "evidence_chain": ["duckdb:9412abc00841", "duckdb:5a7f8d0d5456"],
    }
    assert check_numeric_grounding("The shortfall is 9412 units.", facts) == ["9412"]
    assert check_numeric_grounding("Cover is 1.9 days.", facts) == []


def test_prose_the_model_was_shown_does_not_ground_new_figures() -> None:
    """Reusing a number out of a reason string is not evidence it was computed."""
    facts = {"quantity": 400, "reason": "a 987-unit shortfall was projected"}
    assert check_numeric_grounding("Transfer 400 units.", facts) == []
    assert check_numeric_grounding("The shortfall is 987 units.", facts) == ["987"]


def test_the_same_figure_written_differently_is_still_grounded() -> None:
    facts = {"delay_days": 5, "days_of_cover": 1.8, "revised_eta": "2026-08-26"}
    assert check_numeric_grounding("5.0 days late, 1.80 days of cover.", facts) == []
    assert check_numeric_grounding("ETA 2026-08-26 (day 26 of month 8).", facts) == []


def test_narration_returns_the_ungrounded_numbers_alongside_the_text() -> None:
    planner = _planner([
        SimpleNamespace(id="r9", output=[], output_text="Shortfall is about 1200 units.")
    ])
    from avathon.llm import PlannerRun

    run = PlannerRun(model="test-model")
    narrative, ungrounded = planner.narrate(run=run, facts=FACTS)
    assert narrative == "Shortfall is about 1200 units."
    assert ungrounded == ["1200"]
    assert run.response_ids == ["r9"]
