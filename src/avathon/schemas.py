"""Typed contracts for every handoff in the agent workflow.

The design deliberately keeps operational evidence and machine decisions as
structured data; free-form model text is explanatory, not the system of record.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, TypedDict
from uuid import uuid4


class MessageType(str, Enum):
    TOOL_RESULT = "tool_result"
    RISK_ASSESSMENT = "risk_assessment"
    IMPACT_ASSESSMENT = "impact_assessment"
    ACTION_PROPOSAL = "action_proposal"
    APPROVAL_REQUEST = "approval_request"
    HUMAN_DECISION = "human_decision"
    ESCALATION = "escalation"


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
    """LangGraph state. Only typed messages cross agent boundaries."""

    trace_id: str
    scenario_id: str
    decision: Literal["approve", "reject", "modify", "pending"]
    messages: list[dict[str, Any]]
    risk: dict[str, Any]
    impact: dict[str, Any]
    proposal: dict[str, Any]
    escalation_reason: str

