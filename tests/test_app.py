"""The console must not become a way around the safety boundary."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
TIMEOUT = 60


def _button(app: AppTest, label: str):
    return next(button for button in app.button if button.label == label)


def _start(app: AppTest) -> AppTest:
    app.run(timeout=TIMEOUT)
    return app


def test_running_a_scenario_without_a_key_is_refused_not_silently_downgraded() -> None:
    """LLM-only means LLM-only: there is no hidden deterministic fallback."""
    app = _start(AppTest.from_file(APP_PATH))
    _button(app, "Run scenario").click().run(timeout=TIMEOUT)

    assert not app.exception
    assert any("API key is required" in error.value for error in app.error)
    assert all(button.label != "Approve simulated action" for button in app.button)


def test_the_guardrail_probe_blocks_an_out_of_scope_planner_in_the_ui() -> None:
    app = _start(AppTest.from_file(APP_PATH))
    _button(app, "Run adversarial planner").click().run(timeout=TIMEOUT)

    assert not app.exception
    assert any("Guardrail block" in error.value for error in app.error)
    assert any("out_of_scope_sku" in error.value for error in app.error)
    assert all(button.label != "Approve simulated action" for button in app.button)


def test_the_probe_run_records_no_action_proposal() -> None:
    app = _start(AppTest.from_file(APP_PATH))
    _button(app, "Run adversarial planner").click().run(timeout=TIMEOUT)

    state = app.session_state["run"]
    values = state["graph"].get_state(state["config"]).values
    types = [message["message_type"] for message in values["messages"]]
    assert "action_proposal" not in types
    assert values["escalation_type"] == "guardrail_violation"


def test_the_console_opens_without_a_run_and_asks_for_input() -> None:
    app = _start(AppTest.from_file(APP_PATH))
    assert not app.exception
    assert any("choose a scenario" in info.value.lower() for info in app.info)


def test_each_scenario_states_the_expectation_of_the_run_it_performs() -> None:
    """A scenario reused in the matrix must not display another case's outcome.

    ``verified_delay`` appears twice - once approved, once rejected. The console
    runs the approved path, so that is the expectation it must display.
    """
    from avathon.data_generation import SCENARIOS
    from avathon.evaluation import primary_expectations

    expectations = primary_expectations()
    assert expectations["verified_delay"].expect_escalation is None
    assert expectations["verified_delay"].expect_action == "transfer_stock"

    for scenario, case in expectations.items():
        assert scenario in SCENARIOS
        assert case.planner == "reference"
