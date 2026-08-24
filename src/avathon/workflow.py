"""LangGraph orchestration with a mandatory human approval interrupt."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .agents import (
    AgentOutcome,
    DemandImpactAgent,
    ReplenishmentRecommendationAgent,
    RiskDetectionEvidenceAgent,
)
from .llm import DEFAULT_MODEL, LLMPlanner, Planner
from .schemas import AgentMessage, EscalationType, MessageType, WorkflowState
from .tools import SupplyChainAnalyticsTool


def build_workflow(
    data_dir: Path,
    *,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    planner: Planner | None = None,
    planner_factory=None,
):
    """Return a compiled graph interrupted before human review.

    Exactly one planner source is required: a live ``api_key``, an injected
    ``planner``, or a ``planner_factory`` mapping an agent name to a planner.
    There is no unplanned fallback path — a workflow with no planner cannot run,
    rather than silently degrading into hardcoded behaviour.
    """
    tool = SupplyChainAnalyticsTool(data_dir)

    if planner_factory is not None:
        make = planner_factory
    elif planner is not None:
        def make(_agent: str) -> Planner:
            return planner
    elif api_key:
        def make(_agent: str) -> Planner:
            return LLMPlanner(api_key=api_key, model=model)
    else:
        raise ValueError(
            "build_workflow requires an OpenAI api_key, a planner, or a planner_factory."
        )

    risk_agent = RiskDetectionEvidenceAgent(tool, make("risk_agent"))
    impact_agent = DemandImpactAgent(tool, make("impact_agent"))
    recommendation_agent = ReplenishmentRecommendationAgent(tool, make("replenishment_agent"))

    def _apply(outcome: AgentOutcome, key: str) -> dict:
        result: dict = {"messages": [message.to_dict() for message in outcome.messages]}
        if outcome.escalated:
            result["escalation_reason"] = outcome.escalation_reason
            result["escalation_type"] = outcome.escalation_type
        else:
            result[key] = outcome.payload
        return result

    def risk_node(state: WorkflowState) -> dict:
        outcome = risk_agent.assess(
            state["trace_id"], state["scenario_id"],
            state.get("sku", "SKU-CRITICAL"), state.get("site", "Pune-DC"),
        )
        return _apply(outcome, "risk")

    def impact_node(state: WorkflowState) -> dict:
        return _apply(
            impact_agent.assess(state["trace_id"], state["scenario_id"], state["risk"]),
            "impact",
        )

    def recommendation_node(state: WorkflowState) -> dict:
        return _apply(
            recommendation_agent.propose(
                state["trace_id"], state["scenario_id"], state["impact"]
            ),
            "proposal",
        )

    def route_on_escalation(next_node: str):
        def route(state: WorkflowState) -> Literal["continue", "escalate"]:
            return "escalate" if state.get("escalation_type") else "continue"
        route.__name__ = f"route_to_{next_node}"
        return route

    def human_review_node(state: WorkflowState) -> dict:
        decision = state.get("decision", "pending")
        message = AgentMessage.create(
            trace_id=state["trace_id"], scenario_id=state["scenario_id"],
            sender="human_planner", recipient="workflow",
            message_type=MessageType.HUMAN_DECISION,
            payload={
                "decision": decision,
                "planner_id": state.get("planner_id", "unattributed"),
                "proposal_action": state["proposal"]["action"],
                "proposal_quantity": state["proposal"]["quantity"],
                "proposal_evidence": state["proposal"].get("evidence_chain", []),
            },
            evidence_refs=state["proposal"].get("evidence_chain", []), confidence=1.0,
        )
        update: dict = {"messages": [message.to_dict()]}
        if decision != "approve":
            update["escalation_type"] = (
                EscalationType.HUMAN_REJECTED.value if decision == "reject"
                else EscalationType.HUMAN_NO_DECISION.value
            )
            update["escalation_reason"] = (
                f"Planner '{state.get('planner_id', 'unattributed')}' did not approve "
                f"the proposed {state['proposal']['action']}."
            )
        return update

    def route_after_human(state: WorkflowState) -> Literal["finalize", "escalate"]:
        return "finalize" if state.get("decision") == "approve" else "escalate"

    def finalize_node(state: WorkflowState) -> dict:
        proposal = state["proposal"]
        message = AgentMessage.create(
            trace_id=state["trace_id"], scenario_id=state["scenario_id"],
            sender="workflow", recipient="audit_log",
            message_type=MessageType.ACTION_PROPOSAL,
            payload={
                **proposal,
                "execution_status": "approved_for_simulated_execution",
                "approved_by": state.get("planner_id", "unattributed"),
            },
            evidence_refs=proposal.get("evidence_chain", []), confidence=1.0,
        )
        return {"messages": [message.to_dict()]}

    def escalate_node(state: WorkflowState) -> dict:
        message = AgentMessage.create(
            trace_id=state["trace_id"], scenario_id=state["scenario_id"],
            sender="workflow", recipient="human_planner",
            message_type=MessageType.ESCALATION,
            payload={
                "reason": state.get("escalation_reason", "Workflow stopped without approval."),
                "escalation_type": state.get(
                    "escalation_type", EscalationType.HUMAN_NO_DECISION.value
                ),
                "action_proposed": False,
                "action_executed": False,
            },
            confidence=1.0,
        )
        return {"messages": [message.to_dict()]}

    graph = StateGraph(WorkflowState)
    graph.add_node("risk_agent", risk_node)
    graph.add_node("impact_agent", impact_node)
    graph.add_node("replenishment_agent", recommendation_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("escalate", escalate_node)

    graph.add_edge(START, "risk_agent")
    graph.add_conditional_edges(
        "risk_agent", route_on_escalation("impact_agent"),
        {"continue": "impact_agent", "escalate": "escalate"},
    )
    graph.add_conditional_edges(
        "impact_agent", route_on_escalation("replenishment_agent"),
        {"continue": "replenishment_agent", "escalate": "escalate"},
    )
    graph.add_conditional_edges(
        "replenishment_agent", route_on_escalation("human_review"),
        {"continue": "human_review", "escalate": "escalate"},
    )
    graph.add_conditional_edges(
        "human_review", route_after_human, {"finalize": "finalize", "escalate": "escalate"}
    )
    graph.add_edge("finalize", END)
    graph.add_edge("escalate", END)
    return graph.compile(checkpointer=MemorySaver(), interrupt_before=["human_review"])
