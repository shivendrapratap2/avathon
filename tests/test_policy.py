"""The deterministic core: evidence integrity, impact math, action bounds."""

from __future__ import annotations

import pytest

from avathon import policy


def _update(po_id: str, *, reported: str, eta: str, status: str = "delayed",
            quality: str = "verified", delay: int = 5, on_hand: int = 175,
            safety: int = 500, on_order: int = 0, quantity: int = 500,
            supplier: str = "SUP-A") -> dict:
    return {
        "po_id": po_id, "supplier": supplier, "quantity": quantity,
        "reported_date": reported, "eta_date": eta, "shipment_status": status,
        "source_quality": quality, "delay_days": delay,
        "on_hand": on_hand, "safety_stock": safety, "on_order": on_order,
    }


# -- evidence integrity ------------------------------------------------------


def test_absent_evidence_fails_closed() -> None:
    verdict = policy.assess_evidence([])
    assert not verdict.ok
    assert verdict.escalation_type == "no_evidence"


def test_simultaneous_divergent_updates_for_one_po_are_a_conflict() -> None:
    rows = [
        _update("PO-001", reported="2026-08-22", eta="2026-08-26"),
        _update("PO-001", reported="2026-08-22", eta="2026-08-24",
                status="on_time", quality="unverified"),
    ]
    verdict = policy.assess_evidence(rows)
    assert not verdict.ok
    assert verdict.escalation_type == "conflicting_evidence"
    assert verdict.conflicting_pos == ["PO-001"]


def test_several_open_pos_are_not_a_conflict() -> None:
    """The regression that matters: multiple POs into one site is normal operations."""
    rows = [
        _update("PO-001", reported="2026-08-22", eta="2026-08-26"),
        _update("PO-003", reported="2026-08-22", eta="2026-09-01",
                status="on_time", delay=0, supplier="SUP-B"),
    ]
    verdict = policy.assess_evidence(rows)
    assert verdict.ok
    assert verdict.po_count == 2
    assert len(verdict.effective_rows) == 2


def test_a_later_update_supersedes_an_earlier_one() -> None:
    """Revision is not contradiction: only the latest reported date is considered."""
    rows = [
        _update("PO-001", reported="2026-08-19", eta="2026-08-21", status="on_time", delay=0),
        _update("PO-001", reported="2026-08-22", eta="2026-08-26"),
    ]
    verdict = policy.assess_evidence(rows)
    assert verdict.ok
    assert verdict.effective_rows[0]["eta_date"] == "2026-08-26"


def test_unverified_latest_update_fails_closed_separately_from_conflict() -> None:
    rows = [_update("PO-001", reported="2026-08-22", eta="2026-08-26", quality="unverified")]
    verdict = policy.assess_evidence(rows)
    assert not verdict.ok
    assert verdict.escalation_type == "unverified_evidence"


def test_risk_summary_reports_the_worst_delay_across_pos() -> None:
    rows = [
        _update("PO-001", reported="2026-08-22", eta="2026-08-26", delay=5),
        _update("PO-002", reported="2026-08-22", eta="2026-08-30", delay=9, supplier="SUP-B"),
    ]
    verdict = policy.assess_evidence(rows)
    risk = policy.summarize_risk("SKU-CRITICAL", "Pune-DC", verdict)
    assert risk["delay_days"] == 9
    assert risk["driving_po_id"] == "PO-002"
    assert risk["open_po_count"] == 2
    assert risk["inbound_units"] == 1000


# -- impact ------------------------------------------------------------------


def _risk(on_hand: int = 175, safety: int = 500, delay: int = 5, on_order: int = 0) -> dict:
    return {
        "sku": "SKU-CRITICAL", "site": "Pune-DC", "delayed": True, "delay_days": delay,
        "on_hand": on_hand, "safety_stock": safety, "on_order": on_order,
    }


def test_exposure_is_measured_against_safety_stock_not_bare_on_hand() -> None:
    impact = policy.estimate_impact(_risk(), [100] * 28)
    # 100/day * 5 days = 500 projected, + 500 safety floor - 175 available = 825 short.
    assert impact["projected_delay_demand_units"] == 500
    assert impact["safety_stock_gap_units"] == 825
    assert impact["breaches_safety_stock"] is True
    assert impact["risk_level"] == "high"


def test_a_position_below_safety_stock_is_high_risk_even_with_no_demand() -> None:
    """The old failure: zero demand produced 999 days of cover and a monitor."""
    impact = policy.estimate_impact(_risk(on_hand=175, safety=500), [0] * 28)
    assert impact["days_of_cover"] is None
    assert impact["risk_level"] == "high"
    assert impact["safety_stock_gap_units"] == 325


def test_ample_cover_is_low_risk() -> None:
    impact = policy.estimate_impact(_risk(on_hand=2000), [100] * 28)
    assert impact["risk_level"] == "low"
    assert impact["safety_stock_gap_units"] == 0


def test_thin_headroom_is_medium_not_low() -> None:
    # available 1050: projected 500 + safety 500 = 1000, headroom 50 < 25% of 500.
    impact = policy.estimate_impact(_risk(on_hand=1050), [100] * 28)
    assert impact["risk_level"] == "medium"


def test_inbound_stock_already_on_order_counts_towards_cover() -> None:
    with_order = policy.estimate_impact(_risk(on_order=400), [100] * 28)
    without = policy.estimate_impact(_risk(on_order=0), [100] * 28)
    assert with_order["safety_stock_gap_units"] == without["safety_stock_gap_units"] - 400


def test_volatile_demand_lowers_confidence() -> None:
    steady = policy.estimate_impact(_risk(), [100] * 28)
    volatile = policy.estimate_impact(_risk(), [20, 180] * 14)
    assert volatile["impact_confidence"] < steady["impact_confidence"]


# -- action bounds -----------------------------------------------------------


def _alternates(surplus: int, site: str = "Mumbai-DC") -> list[dict]:
    return [{"site": site, "on_hand": surplus + 500, "safety_stock": 500,
             "on_order": 0, "transferable_surplus": surplus}]


def test_transfer_never_exceeds_real_surplus_at_the_source() -> None:
    impact = policy.estimate_impact(_risk(), [100] * 28)  # 825-unit gap
    proposal = policy.select_action(impact, _alternates(120))
    assert proposal["action"] == "transfer_stock"
    assert proposal["quantity"] == 120
    assert proposal["source_site"] == "Mumbai-DC"
    assert any("surplus" in note for note in proposal["constraints"])


def test_transfer_never_exceeds_the_single_proposal_policy_cap() -> None:
    impact = policy.estimate_impact(_risk(), [100] * 28)
    proposal = policy.select_action(impact, _alternates(5000))
    assert proposal["quantity"] == policy.MAX_TRANSFER_UNITS
    assert any("policy cap" in note for note in proposal["constraints"])


def test_residual_exposure_is_always_handed_back_to_the_planner() -> None:
    impact = policy.estimate_impact(_risk(), [100] * 28)
    proposal = policy.select_action(impact, _alternates(120))
    assert any("units of exposure remain" in note for note in proposal["constraints"])


def test_no_surplus_anywhere_falls_back_to_a_reversible_expedite() -> None:
    impact = policy.estimate_impact(_risk(), [100] * 28)
    proposal = policy.select_action(impact, _alternates(0))
    assert proposal["action"] == "expedite_supplier"
    assert proposal["quantity"] == 0


def test_low_risk_does_not_move_stock() -> None:
    impact = policy.estimate_impact(_risk(on_hand=2000), [100] * 28)
    proposal = policy.select_action(impact, _alternates(5000))
    assert proposal["action"] == "monitor"
    assert proposal["quantity"] == 0


def test_medium_risk_holds_rather_than_moving_stock_prematurely() -> None:
    impact = policy.estimate_impact(_risk(on_hand=1050), [100] * 28)
    proposal = policy.select_action(impact, _alternates(5000))
    assert proposal["action"] == "monitor"


@pytest.mark.parametrize("action", sorted(policy.ROLLBACK))
def test_every_action_carries_its_own_rollback_instruction(action: str) -> None:
    assert policy.ROLLBACK[action]
    assert "transfer" not in policy.ROLLBACK["monitor"].lower()


def test_every_proposal_is_blocked_pending_approval_at_creation() -> None:
    impact = policy.estimate_impact(_risk(), [100] * 28)
    for alternates in (_alternates(0), _alternates(120), _alternates(5000)):
        proposal = policy.select_action(impact, alternates)
        assert proposal["execution_status"] == "blocked_pending_human_approval"


def test_a_proposal_is_never_more_confident_than_its_estimate() -> None:
    impact = policy.estimate_impact(_risk(), [100] * 28)
    assert policy.proposal_confidence(impact) <= impact["impact_confidence"]
