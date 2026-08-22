# Architecture and controls

## Orchestration

The system uses a LangGraph state graph:

```text
START -> Risk Detection & Evidence -> Demand & Impact -> Replenishment Recommendation
                                                          -> [interrupt: human review]
                                                             -> approve -> simulated audit event -> END
                                                             -> reject/no decision -> escalation -> END
Risk or impact evidence failure -------------------------------------------> escalation -> END
```

LangGraph was selected over CrewAI and AutoGen because this workflow needs
explicit state transitions, persisted checkpoints, conditional branching, and
a first-class interruption immediately before an operational action. CrewAI's
role-oriented conversational abstraction is quicker to prototype but less
precise for approval gates. AutoGen is useful for open-ended agent discussion,
but that flexibility is a liability for a safety-sensitive operational path.
The trade-off is more explicit graph code and less free-form collaboration.

## Agent boundaries

1. **Risk Detection & Evidence Agent** calls a parameterized, read-only DuckDB
   tool for delay and inventory evidence. It detects duplicate/conflicting or
   unverified shipment updates.
2. **Demand & Impact Agent** independently retrieves recent demand and applies
   a transparent 28-day mean forecast with standard-deviation uncertainty. It
   estimates days of cover and delay-period demand.
3. **Replenishment Recommendation Agent** transforms only validated impact
   information into a policy-bounded, reversible proposal. It has no database
   write capability and cannot create or modify a purchase order.

The SQL tool exposes only two named operations (`supplier_exposure` and
`demand_history`), clamps the lookback window, uses query parameters for all
agent inputs, and records a query ID. An LLM could generate planner-facing
explanations over these structured outputs, but numerical assessments and
execution permissions never depend on unconstrained model text.

## Message contract

Every cross-node message is an `AgentMessage` defined in
`src/avathon/schemas.py`:

| Field | Type | Purpose |
| --- | --- | --- |
| `trace_id`, `scenario_id` | string | Correlate the complete decision path |
| `sender`, `recipient` | string | Explicit accountability boundary |
| `message_type` | enum | `tool_result`, `risk_assessment`, `impact_assessment`, `approval_request`, `human_decision`, `escalation` |
| `payload` | object | Typed operational finding or proposal |
| `evidence_refs` | string[] | Immutable query IDs or source snapshots |
| `confidence` | float [0,1] | Calibrated decision confidence |
| `assumptions` | string[] | Conditions a reviewer must validate |
| `timestamp_utc` | ISO-8601 string | Audit chronology |

Example:

```json
{
  "trace_id": "trace-success-001",
  "scenario_id": "success",
  "sender": "replenishment_agent",
  "recipient": "human_planner",
  "message_type": "approval_request",
  "payload": {
    "sku": "SKU-CRITICAL",
    "site": "Pune-DC",
    "action": "transfer_stock",
    "quantity": 280,
    "execution_status": "blocked_pending_human_approval",
    "rollback": "Cancel the transfer before dispatch confirmation; no PO is modified."
  },
  "evidence_refs": ["duckdb:example-query-id"],
  "confidence": 0.75,
  "assumptions": ["Alternate DC passes a live safety-stock check."],
  "timestamp_utc": "2026-08-22T00:00:00+00:00"
}
```

## Failure controls and observability

The workflow fails closed on missing, unverified, or contradictory shipment
data; it emits an escalation rather than an action proposal. Other controls
are allow-listed parameterized tool calls, immutable evidence references,
structured message validation, human approval, and a rollback description for
every proposed action. `run_demo.py` writes JSONL traces containing every agent
handoff, source query reference, approval event, and final status.

Production monitoring should track data freshness, tool failure rate,
escalation/approval/rejection rates, planner overrides, latency per node, and
LLM tokens/cost if a narrative model is enabled. Deploy versioned graphs and
policies through replay tests and canaries; use durable checkpoint storage
instead of the demo's in-memory store, so workflow changes can be rolled back
without downtime.
