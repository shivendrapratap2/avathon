"""LangGraph orchestration with a mandatory human approval interrupt."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .agents import DemandImpactAgent, ReplenishmentRecommendationAgent, RiskDetectionEvidenceAgent
from .schemas import AgentMessage, MessageType, WorkflowState
from .tools import SupplyChainAnalyticsTool


def _append(state: WorkflowState, message: AgentMessage) -> dict:
    return {"messages": [*state.get("messages", []), message.to_dict()]}


def build_workflow(data_dir: Path):
    """Return a compiled LangGraph graph interrupted before human review."""
    tool = SupplyChainAnalyticsTool(data_dir)
    risk_agent = RiskDetectionEvidenceAgent(tool)
    impact_agent = DemandImpactAgent(tool)
    recommendation_agent = ReplenishmentRecommendationAgent()

    def risk_node(state: WorkflowState) -> dict:
        msg = risk_agent.assess(
            state["trace_id"], state["scenario_id"], state.get("sku", "SKU-CRITICAL"),
            state.get("site", "Pune-DC"),
        )
        result = _append(state, msg)
        if msg.message_type == MessageType.ESCALATION:
            result["escalation_reason"] = msg.payload["reason"]
        else:
            result["risk"] = msg.payload
        return result

    def route_after_risk(state: WorkflowState) -> Literal["impact_agent", "escalate"]:
        return "escalate" if state.get("escalation_reason") else "impact_agent"

    def impact_node(state: WorkflowState) -> dict:
        msg = impact_agent.assess(state["trace_id"], state["scenario_id"], state["risk"])
        result = _append(state, msg)
        if msg.message_type == MessageType.ESCALATION:
            result["escalation_reason"] = msg.payload["reason"]
        else:
            result["impact"] = msg.payload
        return result

    def route_after_impact(state: WorkflowState) -> Literal["replenishment_agent", "escalate"]:
        return "escalate" if state.get("escalation_reason") else "replenishment_agent"

    def recommendation_node(state: WorkflowState) -> dict:
        msg = recommendation_agent.propose(state["trace_id"], state["scenario_id"], state["impact"])
        result = _append(state, msg)
        result["proposal"] = msg.payload
        return result

    def human_review_node(state: WorkflowState) -> dict:
        decision = state.get("decision", "pending")
        message = AgentMessage.create(
            trace_id=state["trace_id"], scenario_id=state["scenario_id"], sender="human_planner",
            recipient="workflow", message_type=MessageType.HUMAN_DECISION,
            payload={"decision": decision, "proposal": state["proposal"]}, confidence=1.0,
        )
        return _append(state, message)

    def route_after_human(state: WorkflowState) -> Literal["finalize", "escalate"]:
        return "finalize" if state.get("decision") == "approve" else "escalate"

    def finalize_node(state: WorkflowState) -> dict:
        message = AgentMessage.create(
            trace_id=state["trace_id"], scenario_id=state["scenario_id"], sender="workflow",
            recipient="audit_log", message_type=MessageType.ACTION_PROPOSAL,
            payload={**state["proposal"], "execution_status": "approved_for_simulated_execution"},
            confidence=1.0,
        )
        return _append(state, message)

    def escalate_node(state: WorkflowState) -> dict:
        message = AgentMessage.create(
            trace_id=state["trace_id"], scenario_id=state["scenario_id"], sender="workflow",
            recipient="human_planner", message_type=MessageType.ESCALATION,
            payload={"reason": state.get("escalation_reason", "Human did not approve action.")},
            confidence=1.0,
        )
        return _append(state, message)

    graph = StateGraph(WorkflowState)
    graph.add_node("risk_agent", risk_node)
    graph.add_node("impact_agent", impact_node)
    graph.add_node("replenishment_agent", recommendation_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("escalate", escalate_node)
    graph.add_edge(START, "risk_agent")
    graph.add_conditional_edges(
        "risk_agent", route_after_risk, {"impact_agent": "impact_agent", "escalate": "escalate"}
    )
    graph.add_conditional_edges(
        "impact_agent", route_after_impact,
        {"replenishment_agent": "replenishment_agent", "escalate": "escalate"},
    )
    graph.add_edge("replenishment_agent", "human_review")
    graph.add_conditional_edges("human_review", route_after_human, {"finalize": "finalize", "escalate": "escalate"})
    graph.add_edge("finalize", END)
    graph.add_edge("escalate", END)
    return graph.compile(checkpointer=MemorySaver(), interrupt_before=["human_review"])
