"""The boundary between model intent and tool execution."""

from __future__ import annotations

import pytest

from avathon.guardrails import InvestigationScope, ToolCallSafetyError, ToolPolicy


def _policy(**overrides) -> ToolPolicy:
    defaults = dict(
        agent="risk_agent",
        allowed_tools=frozenset({"supplier_exposure", "demand_history"}),
        required_tools=frozenset({"supplier_exposure"}),
        scope=InvestigationScope(sku="SKU-CRITICAL", site="Pune-DC"),
    )
    defaults.update(overrides)
    return ToolPolicy(**defaults)


def test_an_allowed_call_in_scope_passes_and_is_sanitized() -> None:
    clean = _policy().validate(
        "demand_history", {"sku": "SKU-CRITICAL", "site": "Pune-DC", "lookback_days": 28}
    )
    assert clean == {"sku": "SKU-CRITICAL", "site": "Pune-DC", "lookback_days": 28}


def test_a_tool_outside_the_agents_allow_list_is_refused() -> None:
    with pytest.raises(ToolCallSafetyError) as error:
        _policy().validate("alternate_site_availability",
                           {"sku": "SKU-CRITICAL", "exclude_site": "Pune-DC"})
    assert error.value.violation == "tool_not_allowed"


def test_a_call_reaching_a_different_sku_is_refused() -> None:
    with pytest.raises(ToolCallSafetyError) as error:
        _policy().validate("supplier_exposure", {"sku": "SKU-VOLATILE", "site": "Pune-DC"})
    assert error.value.violation == "out_of_scope_sku"


def test_a_call_reaching_a_different_site_is_refused() -> None:
    with pytest.raises(ToolCallSafetyError) as error:
        _policy().validate("supplier_exposure", {"sku": "SKU-CRITICAL", "site": "Mumbai-DC"})
    assert error.value.violation == "out_of_scope_site"


@pytest.mark.parametrize("lookback", [6, 57, 3650, -1])
def test_a_lookback_outside_the_approved_window_is_refused(lookback: int) -> None:
    with pytest.raises(ToolCallSafetyError) as error:
        _policy().validate(
            "demand_history",
            {"sku": "SKU-CRITICAL", "site": "Pune-DC", "lookback_days": lookback},
        )
    assert error.value.violation == "argument_bounds"


@pytest.mark.parametrize("lookback", ["28", 28.5, True, None])
def test_a_non_integer_lookback_is_refused(lookback: object) -> None:
    with pytest.raises(ToolCallSafetyError) as error:
        _policy().validate(
            "demand_history",
            {"sku": "SKU-CRITICAL", "site": "Pune-DC", "lookback_days": lookback},
        )
    assert error.value.violation == "argument_type"


def test_unexpected_arguments_are_refused() -> None:
    with pytest.raises(ToolCallSafetyError) as error:
        _policy().validate(
            "supplier_exposure",
            {"sku": "SKU-CRITICAL", "site": "Pune-DC", "limit": 99999},
        )
    assert error.value.violation == "argument_schema"


def test_missing_arguments_are_refused() -> None:
    with pytest.raises(ToolCallSafetyError) as error:
        _policy().validate("supplier_exposure", {"sku": "SKU-CRITICAL"})
    assert error.value.violation == "argument_schema"


def test_an_identical_repeated_call_is_refused_so_a_planner_cannot_loop() -> None:
    policy = _policy()
    arguments = {"sku": "SKU-CRITICAL", "site": "Pune-DC"}
    policy.validate("supplier_exposure", dict(arguments))
    with pytest.raises(ToolCallSafetyError) as error:
        policy.validate("supplier_exposure", dict(arguments))
    assert error.value.violation == "duplicate_call"


def test_a_differing_repeat_of_the_same_tool_is_allowed() -> None:
    policy = _policy()
    base = {"sku": "SKU-CRITICAL", "site": "Pune-DC"}
    policy.validate("demand_history", {**base, "lookback_days": 14})
    policy.validate("demand_history", {**base, "lookback_days": 56})


def test_finishing_without_required_evidence_fails_closed() -> None:
    with pytest.raises(ToolCallSafetyError) as error:
        _policy().check_required_evidence({"demand_history"})
    assert error.value.violation == "missing_required_evidence"


def test_required_evidence_satisfied_passes() -> None:
    _policy().check_required_evidence({"supplier_exposure", "demand_history"})


def test_only_allow_listed_specs_are_published_to_the_model() -> None:
    names = {spec["name"] for spec in _policy().tool_specs}
    assert names == {"supplier_exposure", "demand_history"}


def test_the_allow_list_is_wider_than_the_requirement() -> None:
    """Tool choice must be a real decision, not a single legal move."""
    policy = _policy()
    assert policy.required_tools < policy.allowed_tools
