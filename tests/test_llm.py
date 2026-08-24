from types import SimpleNamespace

import pytest

from avathon.llm import OpenAIToolCaller, ToolCallSafetyError


class _FakeResponses:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _caller(responses: list[SimpleNamespace]) -> OpenAIToolCaller:
    caller = object.__new__(OpenAIToolCaller)
    caller.model = "test-model"
    caller.client = SimpleNamespace(responses=_FakeResponses(responses))
    return caller


def test_model_selected_tool_is_validated_and_traced() -> None:
    function_call = SimpleNamespace(
        type="function_call", name="supplier_exposure", arguments='{"sku":"SKU-CRITICAL","site":"Pune-DC"}',
        call_id="call-123",
    )
    first = SimpleNamespace(id="resp-1", output=[function_call])
    second = SimpleNamespace(id="resp-2", output_text="Verified delay evidence was retrieved.")
    caller = _caller([first, second])

    trace = caller.call(
        role="Risk Detection and Evidence Agent", expected_tool="supplier_exposure",
        sku="SKU-CRITICAL", site="Pune-DC",
        execute=lambda **args: {"query_id": "duckdb:q-1", "rows": [args]},
    )

    assert trace.tool_name == "supplier_exposure"
    assert trace.result["query_id"] == "duckdb:q-1"
    assert trace.request_id == "resp-1"
    assert trace.follow_up_id == "resp-2"
    first_request = caller.client.responses.calls[0]
    assert first_request["tool_choice"] == "required"
    assert first_request["store"] is False


def test_wrong_model_tool_is_blocked_before_local_execution() -> None:
    function_call = SimpleNamespace(
        type="function_call", name="demand_history",
        arguments='{"sku":"SKU-CRITICAL","site":"Pune-DC","lookback_days":28}', call_id="call-wrong",
    )
    caller = _caller([SimpleNamespace(id="resp-1", output=[function_call])])

    with pytest.raises(ToolCallSafetyError, match="supplier_exposure"):
        caller.call(
            role="Risk Detection and Evidence Agent", expected_tool="supplier_exposure",
            sku="SKU-CRITICAL", site="Pune-DC",
            execute=lambda **_: pytest.fail("Local tool must not run for the wrong model selection."),
        )
