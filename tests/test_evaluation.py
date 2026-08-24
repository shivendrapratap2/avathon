"""The evaluation harness must itself be trustworthy."""

from __future__ import annotations

from pathlib import Path

from avathon import evaluation


def test_the_full_matrix_passes_against_the_reference_plan(tmp_path: Path) -> None:
    results, summary = evaluation.evaluate(tmp_path)

    assert summary["failed"] == 0, [
        (r.case.name, r.failures) for r in results if not r.passed
    ]
    assert summary["missed_escalations"] == 0
    assert summary["false_escalations"] == 0
    assert summary["actions_without_approval"] == 0
    assert summary["proposals_within_bounds"] == summary["cases"]
    assert summary["guardrail_blocks"] == 1


def test_the_matrix_covers_both_stopping_and_proceeding(tmp_path: Path) -> None:
    """A suite that only tests failure paths cannot detect over-escalation."""
    stopping = [case for case in evaluation.CASES if case.expect_escalation]
    proceeding = [case for case in evaluation.CASES if not case.expect_escalation]
    assert len(stopping) >= 3
    assert len(proceeding) >= 3


def test_a_summary_report_is_written_for_review(tmp_path: Path) -> None:
    evaluation.evaluate(tmp_path)
    report = (tmp_path / "results" / "eval_summary.md").read_text()
    assert "Missed escalations" in report
    assert "False escalations" in report
    for case in evaluation.CASES:
        assert case.name in report
    assert (tmp_path / "results" / "eval_summary.json").exists()


def test_a_broken_workflow_is_actually_caught_by_the_harness(
    tmp_path: Path, monkeypatch
) -> None:
    """Guard against a harness that passes because it checks nothing.

    Force every proposal past the policy cap and confirm the bounds check fails.
    """
    import avathon.policy as policy_engine

    real_select = policy_engine.select_action

    def inflated(impact, alternates):
        proposal = real_select(impact, alternates)
        proposal["quantity"] = proposal["policy_cap_units"] + 1
        return proposal

    monkeypatch.setattr(policy_engine, "select_action", inflated)
    _, summary = evaluation.evaluate(tmp_path)
    assert summary["failed"] > 0
    assert summary["proposals_within_bounds"] < summary["cases"]


def test_every_case_documents_what_it_defends() -> None:
    for case in evaluation.CASES:
        assert case.rationale, f"{case.name} has no stated rationale"
