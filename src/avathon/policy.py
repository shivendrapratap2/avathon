"""Deterministic operating policy: evidence integrity, impact math, action bounds.

Nothing in this module calls a model. Every number a planner sees, and every
number in a proposal shown to a human, is produced here from tool results. The
model may explain these figures; it may not produce or alter them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any


# -- named policy constants (documented, not magic numbers) -----------------

#: A single automated proposal may never request more than this many units.
#: Anything larger is a planning decision, not a triage decision, and belongs
#: with a planner rather than in an automated recommendation.
MAX_TRANSFER_UNITS = 600

#: Minimum days of demand history required before an impact estimate is allowed.
MIN_HISTORY_DAYS = 14

#: Headroom below which a non-breaching position is still flagged as thin,
#: expressed as a fraction of safety stock.
THIN_MARGIN_FRACTION = 0.25

VERIFIED = "verified"


# -- evidence integrity ------------------------------------------------------


@dataclass
class EvidenceVerdict:
    """Outcome of grouping shipment updates by purchase order."""

    ok: bool
    reason: str = ""
    escalation_type: str = ""
    effective_rows: list[dict[str, Any]] = field(default_factory=list)
    conflicting_pos: list[str] = field(default_factory=list)
    unverified_pos: list[str] = field(default_factory=list)
    po_count: int = 0


def assess_evidence(rows: list[dict[str, Any]]) -> EvidenceVerdict:
    """Detect contradictory or unverified shipment evidence, per purchase order.

    Conflict is defined *within* a purchase order, between updates that share the
    same reported date. Two different POs legitimately having different statuses
    is the normal operating case and must not escalate; a later update revising
    an earlier one is a normal correction and must not escalate either. Only
    simultaneous, divergent claims about the same shipment are contradictory.
    """
    if not rows:
        return EvidenceVerdict(
            ok=False,
            reason="No purchase-order or shipment evidence exists for this SKU and site.",
            escalation_type="no_evidence",
        )

    by_po: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_po.setdefault(str(row["po_id"]), []).append(row)

    effective: list[dict[str, Any]] = []
    conflicting: list[str] = []
    unverified: list[str] = []

    for po_id, updates in by_po.items():
        latest_date = max(str(update["reported_date"]) for update in updates)
        current = [u for u in updates if str(u["reported_date"]) == latest_date]

        etas = {str(u["eta_date"]) for u in current}
        statuses = {str(u["shipment_status"]) for u in current}
        if len(etas) > 1 or len(statuses) > 1:
            conflicting.append(po_id)
            continue
        if any(str(u["source_quality"]) != VERIFIED for u in current):
            unverified.append(po_id)
            continue
        effective.append(current[0])

    if conflicting:
        return EvidenceVerdict(
            ok=False,
            reason=(
                "Simultaneous contradictory shipment updates for "
                f"{', '.join(sorted(conflicting))}; the true ETA is unknown."
            ),
            escalation_type="conflicting_evidence",
            conflicting_pos=sorted(conflicting),
            unverified_pos=sorted(unverified),
            po_count=len(by_po),
        )
    if unverified:
        return EvidenceVerdict(
            ok=False,
            reason=(
                "Latest shipment update for "
                f"{', '.join(sorted(unverified))} is unverified; provenance is insufficient to act."
            ),
            escalation_type="unverified_evidence",
            unverified_pos=sorted(unverified),
            po_count=len(by_po),
        )

    return EvidenceVerdict(
        ok=True, effective_rows=effective, po_count=len(by_po)
    )


def summarize_risk(sku: str, site: str, verdict: EvidenceVerdict) -> dict[str, Any]:
    """Reduce validated evidence to the risk facts downstream agents rely on."""
    rows = verdict.effective_rows
    delayed = [row for row in rows if int(row["delay_days"]) > 0]
    driver = max(delayed, key=lambda row: int(row["delay_days"])) if delayed else rows[0]
    inbound = sum(int(row["quantity"]) for row in rows)
    return {
        "sku": sku,
        "site": site,
        "delayed": bool(delayed),
        "delay_days": max((int(row["delay_days"]) for row in delayed), default=0),
        "driving_po_id": str(driver["po_id"]),
        "supplier": str(driver["supplier"]),
        "revised_eta": str(driver["eta_date"]),
        "open_po_count": verdict.po_count,
        "delayed_po_count": len(delayed),
        "inbound_units": inbound,
        "on_hand": int(rows[0]["on_hand"]),
        "safety_stock": int(rows[0]["safety_stock"]),
        "on_order": int(rows[0]["on_order"]),
        "evidence_verified": True,
    }


def risk_confidence(risk: dict[str, Any]) -> float:
    """Verified, single-source evidence is trusted; breadth of POs adds noise."""
    base = 0.9
    if risk["open_po_count"] > 1:
        base -= 0.05
    return round(max(base, 0.5), 2)


# -- impact ------------------------------------------------------------------


def estimate_impact(risk: dict[str, Any], units: list[int]) -> dict[str, Any]:
    """Compute exposure over the delay window from on-hand, inbound, and demand.

    Unlike a bare days-of-cover figure, this compares projected demand against
    the position the business is contractually required to hold: safety stock is
    a floor to protect, not buffer to consume.
    """
    daily_mean = round(mean(units), 1)
    dispersion = round(pstdev(units), 1)
    available = risk["on_hand"] + risk["on_order"]
    projected_demand = round(daily_mean * risk["delay_days"])
    gap = projected_demand + risk["safety_stock"] - available
    headroom = -gap
    days_of_cover = round(available / daily_mean, 1) if daily_mean > 0 else None

    if gap > 0:
        risk_level = "high"
    elif headroom < THIN_MARGIN_FRACTION * risk["safety_stock"]:
        risk_level = "medium"
    else:
        risk_level = "low"

    coefficient_of_variation = (dispersion / daily_mean) if daily_mean > 0 else 1.0
    confidence = round(min(0.85, max(0.4, 0.85 - coefficient_of_variation)), 2)

    return {
        **risk,
        "lookback_days": len(units),
        "forecast_daily_units": daily_mean,
        "forecast_stddev_units": dispersion,
        "demand_coefficient_of_variation": round(coefficient_of_variation, 2),
        "available_units": available,
        "projected_delay_demand_units": projected_demand,
        "days_of_cover": days_of_cover,
        "safety_stock_gap_units": max(gap, 0),
        "headroom_units": max(headroom, 0),
        "breaches_safety_stock": gap > 0,
        "risk_level": risk_level,
        "impact_confidence": confidence,
    }


# -- action selection --------------------------------------------------------

ROLLBACK = {
    "transfer_stock": (
        "Cancel the inter-site transfer request before dispatch confirmation. "
        "No purchase order, ERP record, or stock position is modified until dispatch."
    ),
    "expedite_supplier": (
        "Withdraw the expedite enquiry with the supplier. No purchase-order terms, "
        "pricing, or delivery commitments are amended by this proposal."
    ),
    "monitor": (
        "No operational change is proposed. Re-evaluate on the next shipment "
        "status update or demand refresh."
    ),
}


def select_action(impact: dict[str, Any], alternates: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose a bounded, reversible action grounded in real alternate-site surplus."""
    best = max(
        alternates, key=lambda row: int(row["transferable_surplus"]), default=None
    ) if alternates else None
    surplus = int(best["transferable_surplus"]) if best else 0
    source_site = str(best["site"]) if best and surplus > 0 else None

    requirement = impact["safety_stock_gap_units"]

    if impact["risk_level"] == "low":
        action, quantity, reason = (
            "monitor",
            0,
            "Available stock covers projected demand through the revised ETA with "
            "safety stock intact.",
        )
    elif impact["risk_level"] == "medium":
        action, quantity, reason = (
            "monitor",
            0,
            "Safety stock is not breached, but headroom is thin; hold and re-check "
            "on the next demand refresh rather than moving stock prematurely.",
        )
    elif surplus > 0:
        quantity = min(requirement, surplus, MAX_TRANSFER_UNITS)
        action = "transfer_stock"
        reason = (
            f"Projected demand through the revised ETA breaches safety stock by "
            f"{requirement} units. {source_site} holds {surplus} units above its own "
            f"safety stock, so a bounded transfer closes the gap without creating a "
            f"second shortage."
        )
    else:
        quantity = 0
        action = "expedite_supplier"
        reason = (
            f"Projected demand breaches safety stock by {requirement} units and no "
            "alternate site holds transferable surplus; supplier expedite is the only "
            "reversible option."
        )

    constraints = []
    if action == "transfer_stock" and quantity < requirement:
        if surplus <= quantity:
            constraints.append(
                f"Limited by transferable surplus at {source_site} ({surplus} units above "
                "its own safety stock)."
            )
        if MAX_TRANSFER_UNITS <= quantity:
            constraints.append(
                f"Limited by the {MAX_TRANSFER_UNITS}-unit single-proposal policy cap."
            )
        constraints.append(
            f"{requirement - quantity} units of exposure remain after this transfer and "
            "need a planner decision."
        )

    return {
        "sku": impact["sku"],
        "site": impact["site"],
        "action": action,
        "quantity": quantity,
        "source_site": source_site if action == "transfer_stock" else None,
        "risk_level": impact["risk_level"],
        "reason": reason,
        "constraints": constraints,
        "safety_stock_gap_units": requirement,
        "alternate_site_surplus_units": surplus,
        "estimated_days_of_cover": impact["days_of_cover"],
        "policy_cap_units": MAX_TRANSFER_UNITS,
        "rollback": ROLLBACK[action],
        "execution_status": "blocked_pending_human_approval",
    }


def proposal_confidence(impact: dict[str, Any]) -> float:
    """A proposal is never more confident than the estimate underneath it."""
    return round(min(impact["impact_confidence"], 0.8), 2)
