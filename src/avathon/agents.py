"""Three separately accountable supply-chain agents.

Their operational decisions are deterministic and policy-bounded by design.
An LLM may summarize the structured result for a planner, but it never changes
the numerical conclusion, tool call, or execution permissions.
"""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Any

from .schemas import AgentMessage, MessageType
from .tools import SupplyChainAnalyticsTool


class RiskDetectionEvidenceAgent:
    """Retrieves and validates supplier-delay evidence."""

    name = "risk_agent"

    def __init__(self, tool: SupplyChainAnalyticsTool):
        self.tool = tool

    def assess(self, trace_id: str, scenario_id: str, sku: str, site: str) -> AgentMessage:
        result = self.tool.supplier_exposure(sku, site)
        rows = result["rows"]
        if not rows:
            return AgentMessage.create(
                trace_id=trace_id, scenario_id=scenario_id, sender=self.name,
                recipient="impact_agent", message_type=MessageType.ESCALATION,
                payload={"reason": "No purchase-order/shipment evidence for requested SKU/site."},
                evidence_refs=[result["query_id"]], confidence=0.0,
            )
        statuses = {str(row["shipment_status"]) for row in rows}
        quality_ok = all(row["source_quality"] == "verified" for row in rows)
        conflicting = len(statuses) > 1 or len({row["eta_date"] for row in rows}) > 1
        max_delay = max(int(row["delay_days"]) for row in rows)
        payload = {
            "sku": sku,
            "site": site,
            "delayed": max_delay > 0,
            "delay_days": max_delay,
            "source_quality_ok": quality_ok,
            "conflicting_evidence": conflicting,
            "on_hand": int(rows[0]["on_hand"]),
            "safety_stock": int(rows[0]["safety_stock"]),
            "supplier": str(rows[0]["supplier"]),
            "po_id": str(rows[0]["po_id"]),
        }
        confidence = 0.9 if quality_ok and not conflicting else 0.15
        return AgentMessage.create(
            trace_id=trace_id, scenario_id=scenario_id, sender=self.name,
            recipient="impact_agent", message_type=MessageType.RISK_ASSESSMENT,
            payload=payload, evidence_refs=[result["query_id"]], confidence=confidence,
            assumptions=["Shipment updates are current as of their reported timestamp."],
        )


class DemandImpactAgent:
    """Uses an independent read-only demand query and a transparent forecast."""

    name = "impact_agent"

    def __init__(self, tool: SupplyChainAnalyticsTool):
        self.tool = tool

    def assess(self, trace_id: str, scenario_id: str, risk: dict[str, Any]) -> AgentMessage:
        if risk["conflicting_evidence"] or not risk["source_quality_ok"]:
            return AgentMessage.create(
                trace_id=trace_id, scenario_id=scenario_id, sender=self.name,
                recipient="replenishment_agent", message_type=MessageType.ESCALATION,
                payload={"reason": "Supplier evidence is conflicting or unverified; impact is unsafe to estimate."},
                confidence=0.0,
            )
        result = self.tool.demand_history(risk["sku"], risk["site"])
        units = [int(row["units"]) for row in result["rows"]]
        if len(units) < 14:
            return AgentMessage.create(
                trace_id=trace_id, scenario_id=scenario_id, sender=self.name,
                recipient="replenishment_agent", message_type=MessageType.ESCALATION,
                payload={"reason": "Insufficient demand history for a bounded forecast."},
                evidence_refs=[result["query_id"]], confidence=0.0,
            )
        daily_mean = round(mean(units), 1)
        uncertainty = round(pstdev(units), 1)
        days_of_cover = round(risk["on_hand"] / daily_mean, 1) if daily_mean else 999.0
        delay_demand = round(daily_mean * risk["delay_days"])
        risk_level = "high" if days_of_cover < risk["delay_days"] else "medium"
        return AgentMessage.create(
            trace_id=trace_id, scenario_id=scenario_id, sender=self.name,
            recipient="replenishment_agent", message_type=MessageType.IMPACT_ASSESSMENT,
            payload={
                **risk,
                "forecast_daily_units": daily_mean,
                "forecast_stddev_units": uncertainty,
                "days_of_cover": days_of_cover,
                "delay_demand_units": delay_demand,
                "risk_level": risk_level,
            },
            evidence_refs=[result["query_id"]], confidence=0.78,
            assumptions=["Recent 28-day demand is representative of the next supplier-delay window."],
        )


class ReplenishmentRecommendationAgent:
    """Applies explicit policy bounds; it cannot execute ERP changes."""

    name = "replenishment_agent"

    def propose(self, trace_id: str, scenario_id: str, impact: dict[str, Any]) -> AgentMessage:
        if impact["risk_level"] != "high":
            action = "monitor"
            quantity = 0
            reason = "Inventory cover exceeds the estimated supplier delay."
        else:
            # Bounded transfer is deliberately below the known alternate-site surplus.
            quantity = min(280, max(0, impact["delay_demand_units"] - impact["on_hand"]))
            action = "transfer_stock" if quantity else "expedite_supplier"
            reason = (
                "Projected inventory cover is shorter than the verified delay; request a "
                "bounded inter-DC transfer pending planner approval."
            )
        return AgentMessage.create(
            trace_id=trace_id, scenario_id=scenario_id, sender=self.name,
            recipient="human_planner", message_type=MessageType.APPROVAL_REQUEST,
            payload={
                "sku": impact["sku"], "site": impact["site"], "action": action,
                "quantity": quantity, "risk_level": impact["risk_level"], "reason": reason,
                "estimated_days_of_cover": impact["days_of_cover"],
                "rollback": "Cancel the transfer before dispatch confirmation; no PO is modified.",
                "execution_status": "blocked_pending_human_approval",
            },
            confidence=min(impact["confidence"] if "confidence" in impact else 0.75, 0.75),
            assumptions=["Alternate DC inventory must pass a live safety-stock check before execution."],
        )
