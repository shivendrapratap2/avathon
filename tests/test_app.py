from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _button(app: AppTest, label: str):
    return next(button for button in app.button if button.label == label)


def test_verified_delay_ui_requires_and_records_human_approval() -> None:
    app = AppTest.from_file(APP_PATH)
    app.run(timeout=30)

    _button(app, "Run selected scenario").click().run(timeout=30)
    assert not app.exception
    assert any("Human checkpoint reached" in warning.value for warning in app.warning)
    assert any(button.label == "Approve simulated action" for button in app.button)

    _button(app, "Approve simulated action").click().run(timeout=30)
    assert not app.exception
    assert any("Human approval recorded" in success.value for success in app.success)


def test_conflicting_evidence_ui_fails_closed() -> None:
    app = AppTest.from_file(APP_PATH)
    app.run(timeout=30)
    app.radio[0].set_value("Conflicting shipment data: fail closed").run(timeout=30)

    _button(app, "Run selected scenario").click().run(timeout=30)
    assert not app.exception
    assert any("Workflow escalated safely" in error.value for error in app.error)
    assert all(button.label != "Approve simulated action" for button in app.button)
