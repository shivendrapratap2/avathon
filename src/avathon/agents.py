"""Three separately accountable agents, each a planner inside a policy boundary.

Every agent follows the same shape: the model plans and gathers evidence, the
deterministic policy engine turns that evidence into figures, and the model is
then asked to explain those figures without being able to change them.

Accountability is per agent: each owns its own tool allow-list, its own required
evidence, and its own escalation authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import policy as policy_engine
from .guardrails import InvestigationScope, ToolCallSafetyError, ToolPolicy
from .llm import Planner, PlannerRun
from .schemas import AgentMessage, EscalationType, MessageType
from .tools import SupplyChainAnalyticsTool


@dataclass
class AgentOutcome:
    """What a node produced: the audit events, plus either a payload or a stop."""

    messages: list[AgentMessage] = field(default_factory=list)
    payload: dict[str, Any] | None = None
    escalation_reason: str = ""
    escalation_type: str = ""

    @property
    def escalated(self) -> bool:
        return bool(self.escalation_type)


class _BaseAgent:
    name = "agent"
    allowed_tools: frozenset[str] = frozenset()
    required_tools: frozenset[str] = frozenset()
    max_steps = 4

    def __init__(self, tool: SupplyChainAnalyticsTool, planner: Planner):
        self.tool = tool
        self.planner = planner

    # -- shared machinery --------------------------------------------------

    def _policy(self, sku: str, site: str) -> ToolPolicy:
        return ToolPolicy(
            agent=self.name,
            allowed_tools=self.allowed_tools,
            required_tools=self.required_tools,
            scope=InvestigationScope(sku=sku, site=site),
            max_steps=self.max_steps,
        )

    def _plan(
        self, trace_id: str, scenario_id: str, objective: str, sku: str, site: str
    ) -> tuple[PlannerRun | None, list[AgentMessage], AgentOutcome | None]:
        """Run the planner. Returns (run, audit messages, stop-outcome-or-None)."""
        tool_policy = self._policy(sku, site)
        try:
            run = self.planner.gather(
                objective=objective, policy=tool_policy, execute=self.tool.execute
            )
        except ToolCallSafetyError as error:
            escalation_type = (
                EscalationType.MISSING_REQUIRED_EVIDENCE.value
                if error.violation == "missing_required_evidence"
                else EscalationType.GUARDRAIL_VIOLATION.value
            )
            block = AgentMessage.create(
                trace_id=trace_id, scenario_id=scenario_id, sender=self.name,
                recipient="guardrail", message_type=MessageType.GUARDRAIL_BLOCK,
                payload={
                    "selected_by": self.planner.model,
                    "tool_name": error.tool_name,
                    "violation": error.violation,
                    "detail": str(error),
                    "executed": False,
                },
                confidence=1.0,
            )
            return None, [block], AgentOutcome(
                messages=[block],
                escalation_reason=f"Blocked before execution: {error}",
                escalation_type=escalation_type,
            )
        except Exception as error:  # tool or transport failure
            failure = AgentMessage.create(
                trace_id=trace_id, scenario_id=scenario_id, sender=self.name,
                recipient="workflow", message_type=MessageType.TOOL_RESULT,
                payload={
                    "selected_by": self.planner.model, "executed_by": "local_duckdb",
                    "status": "failed", "error_type": type(error).__name__,
                    "detail": str(error)[:300],
                },
                confidence=1.0,
            )
            return None, [failure], AgentOutcome(
                messages=[failure],
                escalation_reason=f"Evidence retrieval stopped safely: {type(error).__name__}.",
                escalation_type=EscalationType.TOOL_FAILURE.value,
            )

        return run, self._audit(trace_id, scenario_id, run), None

    def _audit(self, trace_id: str, scenario_id: str, run: PlannerRun) -> list[AgentMessage]:
        """Turn a planner run into replayable audit events."""
        messages = [
            AgentMessage.create(
                trace_id=trace_id, scenario_id=scenario_id, sender=self.name,
                recipient="audit_log", message_type=MessageType.PLANNER_STEP,
                payload={
                    "model": run.model,
                    "response_ids": run.response_ids,
                    "tool_calls": run.tool_call_count,
                    "tools_selected": sorted(run.called_tools),
                    "steps": [step.to_dict() for step in run.steps],
                },
                narrative=run.closing_statement, confidence=1.0,
            )
        ]
        for tool_name, result in run.evidence.items():
            messages.append(
                AgentMessage.create(
                    trace_id=trace_id, scenario_id=scenario_id, sender=self.name,
                    recipient=tool_name, message_type=MessageType.TOOL_RESULT,
                    payload={
                        "selected_by": run.model,
                        "executed_by": "local_duckdb",
                        "tool_name": tool_name,
                        "arguments": self._arguments_for(run, tool_name),
                        "query_id": result["query_id"],
                        "row_count": len(result["rows"]),
                    },
                    evidence_refs=[result["query_id"]], confidence=1.0,
                )
            )
        return messages

    @staticmethod
    def _arguments_for(run: PlannerRun, tool_name: str) -> dict[str, Any]:
        for step in reversed(run.steps):
            if step.kind == "tool_result" and step.tool_name == tool_name:
                return step.arguments
        return {}

    def _narrate(self, run: PlannerRun, facts: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Ask for an explanation, then verify every number in it is grounded."""
        try:
            narrative, ungrounded = self.planner.narrate(run=run, facts=facts)
        except Exception as error:  # narration is never load-bearing
            return "", {"ok": False, "error": type(error).__name__}
        if ungrounded:
            return "", {"ok": False, "ungrounded_numbers": ungrounded, "suppressed": True}
        return narrative, {"ok": True}

    def _stop(
        self, trace_id: str, scenario_id: str, recipient: str, reason: str,
        escalation_type: EscalationType, prior: list[AgentMessage],
        evidence_refs: list[str] | None = None,
    ) -> AgentOutcome:
        message = AgentMessage.create(
            trace_id=trace_id, scenario_id=scenario_id, sender=self.name,
            recipient=recipient, message_type=MessageType.ESCALATION,
            payload={"reason": reason, "escalation_type": escalation_type.value},
            evidence_refs=evidence_refs or [], confidence=0.0,
        )
        return AgentOutcome(
            messages=[*prior, message], escalation_reason=reason,
            escalation_type=escalation_type.value,
        )


class RiskDetectionEvidenceAgent(_BaseAgent):
    """Establishes whether a delay is real and whether the evidence can be trusted."""

    name = "risk_agent"
    allowed_tools = frozenset({"supplier_exposure", "demand_history"})
    required_tools = frozenset({"supplier_exposure"})

    def assess(self, trace_id: str, scenario_id: str, sku: str, site: str) -> AgentOutcome:
        objective = (
            f"A supplier-delay alert has been raised for SKU '{sku}' at site '{site}'. "
            "Establish two things: whether any inbound purchase order is running late, "
            "and whether the shipment evidence is internally consistent and verified. "
            "Gather whatever read-only evidence you need to answer that."
        )
        run, messages, stop = self._plan(trace_id, scenario_id, objective, sku, site)
        if stop:
            return stop

        rows = run.evidence["supplier_exposure"]["rows"]
        query_id = run.evidence["supplier_exposure"]["query_id"]
        verdict = policy_engine.assess_evidence(rows)
        if not verdict.ok:
            return self._stop(
                trace_id, scenario_id, "human_planner", verdict.reason,
                EscalationType(verdict.escalation_type), messages, [query_id],
            )

        payload = policy_engine.summarize_risk(sku, site, verdict)
        payload["evidence_chain"] = [query_id]
        narrative, grounding = self._narrate(run, payload)
        payload["narrative_grounding"] = grounding
        messages.append(
            AgentMessage.create(
                trace_id=trace_id, scenario_id=scenario_id, sender=self.name,
                recipient="impact_agent", message_type=MessageType.RISK_ASSESSMENT,
                payload=payload, evidence_refs=[query_id],
                confidence=policy_engine.risk_confidence(payload), narrative=narrative,
                assumptions=[
                    "Shipment updates are current as of their reported timestamp.",
                    "The inventory snapshot reflects stock physically available to allocate.",
                ],
            )
        )
        return AgentOutcome(messages=messages, payload=payload)


class DemandImpactAgent(_BaseAgent):
    """Independently retrieves demand and quantifies exposure over the delay window."""

    name = "impact_agent"
    allowed_tools = frozenset({"demand_history", "supplier_exposure"})
    required_tools = frozenset({"demand_history"})

    def assess(self, trace_id: str, scenario_id: str, risk: dict[str, Any]) -> AgentOutcome:
        sku, site = risk["sku"], risk["site"]
        objective = (
            f"A verified {risk['delay_days']}-day supplier delay affects SKU '{sku}' at "
            f"site '{site}'. Retrieve recent daily demand so the exposure over the delay "
            "window can be computed. Choose a lookback window that suits how volatile "
            "this SKU looks; the approved range is 7 to 56 days."
        )
        run, messages, stop = self._plan(trace_id, scenario_id, objective, sku, site)
        if stop:
            return stop

        result = run.evidence["demand_history"]
        units = [int(row["units"]) for row in result["rows"]]
        if len(units) < policy_engine.MIN_HISTORY_DAYS:
            return self._stop(
                trace_id, scenario_id, "human_planner",
                f"Only {len(units)} days of demand history are available; "
                f"{policy_engine.MIN_HISTORY_DAYS} are required for a bounded estimate.",
                EscalationType.INSUFFICIENT_HISTORY, messages, [result["query_id"]],
            )

        payload = policy_engine.estimate_impact(risk, units)
        payload["evidence_chain"] = [*risk.get("evidence_chain", []), result["query_id"]]
        narrative, grounding = self._narrate(run, payload)
        payload["narrative_grounding"] = grounding
        messages.append(
            AgentMessage.create(
                trace_id=trace_id, scenario_id=scenario_id, sender=self.name,
                recipient="replenishment_agent", message_type=MessageType.IMPACT_ASSESSMENT,
                payload=payload, evidence_refs=[result["query_id"]],
                confidence=payload["impact_confidence"], narrative=narrative,
                assumptions=[
                    f"The last {payload['lookback_days']} days of demand represent the "
                    "delay window.",
                    "Safety stock is a floor to protect, not buffer available to consume.",
                ],
            )
        )
        return AgentOutcome(messages=messages, payload=payload)


class ReplenishmentRecommendationAgent(_BaseAgent):
    """Proposes a bounded, reversible action. It cannot execute an ERP change."""

    name = "replenishment_agent"
    allowed_tools = frozenset({"alternate_site_availability", "supplier_exposure"})
    required_tools = frozenset({"alternate_site_availability"})
    max_steps = 3

    def propose(self, trace_id: str, scenario_id: str, impact: dict[str, Any]) -> AgentOutcome:
        sku, site = impact["sku"], impact["site"]
        objective = (
            f"SKU '{sku}' at site '{site}' is exposed over a "
            f"{impact['delay_days']}-day supplier delay. Before any replenishment option "
            "can be considered, establish what stock other sites hold above their own "
            "safety stock. Retrieve that evidence. Do not propose an action."
        )
        run, messages, stop = self._plan(trace_id, scenario_id, objective, sku, site)
        if stop:
            return stop

        result = run.evidence["alternate_site_availability"]
        payload = policy_engine.select_action(impact, result["rows"])
        evidence_refs = [*impact.get("evidence_chain", []), result["query_id"]]
        payload["evidence_chain"] = evidence_refs
        narrative, grounding = self._narrate(run, payload)
        payload["narrative_grounding"] = grounding

        messages.append(
            AgentMessage.create(
                trace_id=trace_id, scenario_id=scenario_id, sender=self.name,
                recipient="human_planner", message_type=MessageType.APPROVAL_REQUEST,
                payload=payload, evidence_refs=evidence_refs,
                confidence=policy_engine.proposal_confidence(impact), narrative=narrative,
                assumptions=[
                    "Alternate-site stock is unreserved and passes a live availability "
                    "check at execution time.",
                    "No action is taken by this system; a planner authorizes execution.",
                ],
            )
        )
        return AgentOutcome(messages=messages, payload=payload)
