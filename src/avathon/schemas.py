"""Typed contracts for every handoff in the agent workflow.

Operational evidence and machine decisions stay structured. Model-generated
text is explanatory only and is never the system of record: it travels in a
dedicated ``narrative`` field and is never read back by downstream logic.
"""

from __future__ import annotations

import operator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal, TypedDict
from uuid import uuid4


class MessageType(str, Enum):
    PLANNER_STEP = "planner_step"
    TOOL_RESULT = "tool_result"
    GUARDRAIL_BLOCK = "guardrail_block"
    RISK_ASSESSMENT = "risk_assessment"
    IMPACT_ASSESSMENT = "impact_assessment"
    APPROVAL_REQUEST = "approval_request"
    HUMAN_DECISION = "human_decision"
    ACTION_PROPOSAL = "action_proposal"
    ESCALATION = "escalation"


class EscalationType(str, Enum):
    """Why the workflow stopped. Audit consumers must never string-match reasons."""

    NO_EVIDENCE = "no_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    UNVERIFIED_EVIDENCE = "unverified_evidence"
    INSUFFICIENT_HISTORY = "insufficient_history"
    MISSING_REQUIRED_EVIDENCE = "missing_required_evidence"
    TOOL_FAILURE = "tool_failure"
    GUARDRAIL_VIOLATION = "guardrail_violation"
    HUMAN_REJECTED = "human_rejected"
    HUMAN_NO_DECISION = "human_no_decision"


@dataclass(frozen=True)
class AgentMessage:
    """An immutable, serializable message exchanged between workflow nodes."""

    trace_id: str
    scenario_id: str
    sender: str
    recipient: str
    message_type: MessageType
    payload: dict[str, Any]
    evidence_refs: list[str] = field(default_factory=list)
    confidence: float = 0.0
    assumptions: list[str] = field(default_factory=list)
    narrative: str = ""
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["message_type"] = self.message_type.value
        return result

    @classmethod
    def create(cls, **kwargs: Any) -> "AgentMessage":
        kwargs.setdefault("trace_id", f"tr-{uuid4().hex[:12]}")
        return cls(**kwargs)


class WorkflowState(TypedDict, total=False):
    """LangGraph state. Only typed messages cross agent boundaries.

    ``messages`` uses an additive reducer so each node returns only the events
    it produced; nodes never rebuild the full history.
    """

    trace_id: str
    scenario_id: str
    sku: str
    site: str
    planner_id: str
    decision: Literal["approve", "reject", "modify", "pending"]
    messages: Annotated[list[dict[str, Any]], operator.add]
    risk: dict[str, Any]
    impact: dict[str, Any]
    proposal: dict[str, Any]
    escalation_reason: str
    escalation_type: str
