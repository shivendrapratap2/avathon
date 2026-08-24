"""Three separately accountable supply-chain agents.

Their operational decisions are deterministic and policy-bounded by design.
An LLM may summarize the structured result for a planner, but it never changes
the numerical conclusion, tool call, or execution permissions.
"""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Any

from .llm import OpenAIToolCaller, ToolCallSafetyError
from .schemas import AgentMessage, MessageType
from .tools import SupplyChainAnalyticsTool


class RiskDetectionEvidenceAgent:
    """Retrieves and validates supplier-delay evidence."""

    name = "risk_agent"

    def __init__(self, tool: SupplyChainAnalyticsTool, llm: OpenAIToolCaller | None = None):
        self.tool = tool
        self.llm = llm

    def assess(self, trace_id: str, scenario_id: str, sku: str, site: str) -> list[AgentMessage]:
        messages: list[AgentMessage] = []
        try:
            if self.llm:
                trace = self.llm.call(
                    role="Risk Detection and Evidence Agent", expected_tool="supplier_exposure",
                    sku=sku, site=site, execute=self.tool.supplier_exposure,
                )
                result = trace.result
                messages.append(_llm_tool_message(trace_id, scenario_id, self.name, trace))
            else:
                result = self.tool.supplier_exposure(sku, site)
                messages.append(_local_tool_message(
                    trace_id, scenario_id, self.name, "supplier_exposure", {"sku": sku, "site": site}, result
                ))
        except Exception as error:
            messages.append(_tool_error_message(trace_id, scenario_id, self.name, "supplier_exposure", error))
            messages.append(AgentMessage.create(
                trace_id=trace_id, scenario_id=scenario_id, sender=self.name, recipient="human_planner",
                message_type=MessageType.ESCALATION,
                payload={"reason": f"Evidence retrieval stopped safely: {type(error).__name__}."}, confidence=0.0,
            ))
            return messages
        rows = result["rows"]
        if not rows:
            messages.append(AgentMessage.create(
                trace_id=trace_id, scenario_id=scenario_id, sender=self.name,
                recipient="impact_agent", message_type=MessageType.ESCALATION,
                payload={"reason": "No purchase-order/shipment evidence for requested SKU/site."},
                evidence_refs=[result["query_id"]], confidence=0.0,
            ))
            return messages
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
        messages.append(AgentMessage.create(
            trace_id=trace_id, scenario_id=scenario_id, sender=self.name,
            recipient="impact_agent", message_type=MessageType.RISK_ASSESSMENT,
            payload=payload, evidence_refs=[result["query_id"]], confidence=confidence,
            assumptions=["Shipment updates are current as of their reported timestamp."],
        ))
        return messages


class DemandImpactAgent:
    """Uses an independent read-only demand query and a transparent forecast."""

    name = "impact_agent"

    def __init__(self, tool: SupplyChainAnalyticsTool, llm: OpenAIToolCaller | None = None):
        self.tool = tool
        self.llm = llm

    def assess(self, trace_id: str, scenario_id: str, risk: dict[str, Any]) -> list[AgentMessage]:
        messages: list[AgentMessage] = []
        if risk["conflicting_evidence"] or not risk["source_quality_ok"]:
            messages.append(AgentMessage.create(
                trace_id=trace_id, scenario_id=scenario_id, sender=self.name,
                recipient="replenishment_agent", message_type=MessageType.ESCALATION,
                payload={"reason": "Supplier evidence is conflicting or unverified; impact is unsafe to estimate."},
                confidence=0.0,
            ))
            return messages
        try:
            if self.llm:
                trace = self.llm.call(
                    role="Demand and Impact Agent", expected_tool="demand_history", sku=risk["sku"],
                    site=risk["site"], execute=lambda **args: self.tool.demand_history(**args),
                )
                result = trace.result
                messages.append(_llm_tool_message(trace_id, scenario_id, self.name, trace))
            else:
                result = self.tool.demand_history(risk["sku"], risk["site"])
                messages.append(_local_tool_message(
                    trace_id, scenario_id, self.name, "demand_history",
                    {"sku": risk["sku"], "site": risk["site"], "lookback_days": 28}, result,
                ))
        except Exception as error:
            messages.append(_tool_error_message(trace_id, scenario_id, self.name, "demand_history", error))
            messages.append(AgentMessage.create(
                trace_id=trace_id, scenario_id=scenario_id, sender=self.name, recipient="human_planner",
                message_type=MessageType.ESCALATION,
                payload={"reason": f"Impact retrieval stopped safely: {type(error).__name__}."}, confidence=0.0,
            ))
            return messages
        units = [int(row["units"]) for row in result["rows"]]
        if len(units) < 14:
            messages.append(AgentMessage.create(
                trace_id=trace_id, scenario_id=scenario_id, sender=self.name,
                recipient="replenishment_agent", message_type=MessageType.ESCALATION,
                payload={"reason": "Insufficient demand history for a bounded forecast."},
                evidence_refs=[result["query_id"]], confidence=0.0,
            ))
            return messages
        daily_mean = round(mean(units), 1)
        uncertainty = round(pstdev(units), 1)
        days_of_cover = round(risk["on_hand"] / daily_mean, 1) if daily_mean else 999.0
        delay_demand = round(daily_mean * risk["delay_days"])
        risk_level = "high" if days_of_cover < risk["delay_days"] else "medium"
        messages.append(AgentMessage.create(
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
        ))
        return messages


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


def _local_tool_message(
    trace_id: str, scenario_id: str, sender: str, tool_name: str, arguments: dict[str, Any], result: dict[str, Any]
) -> AgentMessage:
    return AgentMessage.create(
        trace_id=trace_id, scenario_id=scenario_id, sender=sender, recipient=tool_name,
        message_type=MessageType.TOOL_RESULT,
        payload={"provider": "local_duckdb", "tool_name": tool_name, "arguments": arguments,
                 "query_id": result["query_id"], "row_count": len(result["rows"])},
        evidence_refs=[result["query_id"]], confidence=1.0,
    )


def _llm_tool_message(trace_id: str, scenario_id: str, sender: str, trace: Any) -> AgentMessage:
    return AgentMessage.create(
        trace_id=trace_id, scenario_id=scenario_id, sender=sender, recipient=trace.tool_name,
        message_type=MessageType.TOOL_RESULT,
        payload={
            "provider": "openai_responses_api", "model": trace.model, "tool_name": trace.tool_name,
            "arguments": trace.arguments, "call_id": trace.call_id, "request_id": trace.request_id,
            "follow_up_id": trace.follow_up_id, "query_id": trace.result["query_id"],
            "row_count": len(trace.result["rows"]), "model_summary": trace.summary,
        },
        evidence_refs=[trace.result["query_id"]], confidence=1.0,
    )


def _tool_error_message(
    trace_id: str, scenario_id: str, sender: str, tool_name: str, error: Exception
) -> AgentMessage:
    return AgentMessage.create(
        trace_id=trace_id, scenario_id=scenario_id, sender=sender, recipient=tool_name,
        message_type=MessageType.TOOL_RESULT,
        payload={"provider": "openai_responses_api", "tool_name": tool_name, "status": "blocked",
                 "error_type": type(error).__name__}, confidence=1.0,
    )
