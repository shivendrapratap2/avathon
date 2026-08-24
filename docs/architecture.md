# Architecture and controls

## The central design decision

An agent that only executes a fixed script is not an agent, and a model that
decides operational quantities is not safe. This system splits the two:

| Concern | Owner | Why |
| --- | --- | --- |
| Which evidence to gather, in what order, with what lookback | **GPT planner** | Genuine judgement under uncertainty; the allow-list is wider than the minimum |
| Whether a tool call may run at all | **Guardrail layer** | Model intent is untrusted input until validated |
| Every number: cover, gap, quantity, risk level | **Policy engine** | Reproducible, inspectable, challengeable by a planner |
| Whether anything happens | **Human** | Nothing executes without a named approval |
| Explaining the result | **GPT narrator** | Fluency helps a planner; it is checked for numeric grounding first |

The model can therefore be wrong about *how to investigate* — a recoverable,
observable error — and cannot be wrong about *what to do*.

## Orchestration

```text
START
  -> Risk Detection & Evidence     (plans queries; validates evidence integrity)
       |-- untrustworthy evidence ------------------------> escalate -> END
  -> Demand & Impact               (chooses lookback; computes exposure)
       |-- insufficient history --------------------------> escalate -> END
  -> Replenishment Recommendation  (must ground transfer in real surplus)
       |-- guardrail violation ---------------------------> escalate -> END
  -> [interrupt: human review]
       |-- approve -> simulated audit event ---------------------------> END
       |-- reject / no decision --------------------------> escalate -> END
```

LangGraph was selected over CrewAI and AutoGen because this workflow needs
explicit state transitions, persisted checkpoints, conditional branching, and a
first-class interruption immediately before an operational action. CrewAI's
role-oriented conversational abstraction is quicker to prototype but less precise
for approval gates. AutoGen is useful for open-ended agent discussion, but that
flexibility is a liability on a safety-sensitive operational path. The trade-off
is more explicit graph code and less free-form collaboration.

`messages` uses an additive reducer, so each node returns only the events it
produced rather than rebuilding the history — a node cannot silently drop another
node's audit events.

## Agent boundaries

Each agent owns its own allow-list, its own *required* evidence, and its own
escalation authority. The allow-list is deliberately wider than the requirement
so that tool selection is a real decision rather than a single legal move.

| Agent | May call | Must obtain | Escalates on |
| --- | --- | --- | --- |
| `risk_agent` | `supplier_exposure`, `demand_history` | `supplier_exposure` | absent, contradictory, or unverified evidence |
| `impact_agent` | `demand_history`, `supplier_exposure` | `demand_history` | fewer than 14 days of history |
| `replenishment_agent` | `alternate_site_availability`, `supplier_exposure` | `alternate_site_availability` | guardrail violation |

The replenishment agent has no database write capability and cannot create or
modify a purchase order. Its proposal is data, not an instruction.

## The guardrail layer

Every model-proposed call passes `guardrails.ToolPolicy.validate` before DuckDB
sees it:

| Check | Violation code | Prevents |
| --- | --- | --- |
| Tool is on this agent's allow-list | `tool_not_allowed` | An agent exceeding its remit |
| Arguments match the schema exactly | `argument_schema` | Injected extra parameters |
| Types are correct | `argument_type` | `lookback_days: "all"` |
| Values are in range | `argument_bounds` | A 10-year scan |
| SKU and site match the investigation | `out_of_scope_sku` / `out_of_scope_site` | Reaching another SKU's or site's data |
| Not an identical repeat | `duplicate_call` | A confused planner looping on spend |
| Required evidence was obtained | `missing_required_evidence` | Concluding without looking |

A violation raises before execution and routes to a `guardrail_violation`
escalation. There is no retry and no degraded path: the workflow never falls back
to producing a recommendation after its planner misbehaved.

Two further properties: the SQL tool exposes only three named, parameterized
operations and never accepts agent-supplied SQL; and conversation state is held
locally rather than through `previous_response_id`, so `store=False` holds end to
end and no operational evidence is retained on the provider side.

## Grounding the explanation

Constraining the decision path is not sufficient. A fluent, wrong number in a
planner-facing summary is precisely the failure that erodes trust in an assistive
system. After the policy engine computes the facts, the narrator is asked for a
two-to-three sentence explanation, and every numeric token in its reply is
checked against the computed facts. An ungrounded number suppresses the entire
narrative and records `narrative_grounding: {ok: false, ungrounded_numbers: [...]}`
in the trace. The decision is unaffected, because the decision never depended on
the text.

## Message contract

Every cross-node message is an `AgentMessage` (`src/avathon/schemas.py`):

| Field | Type | Purpose |
| --- | --- | --- |
| `trace_id`, `scenario_id` | string | Correlate the complete decision path |
| `sender`, `recipient` | string | Explicit accountability boundary |
| `message_type` | enum | `planner_step`, `tool_result`, `guardrail_block`, `risk_assessment`, `impact_assessment`, `approval_request`, `human_decision`, `action_proposal`, `escalation` |
| `payload` | object | Typed operational finding or proposal |
| `evidence_refs` | string[] | Immutable query IDs |
| `confidence` | float [0,1] | Decision confidence, derived from evidence quality and demand dispersion |
| `assumptions` | string[] | Conditions a reviewer must validate |
| `narrative` | string | Model-written explanation, grounding-checked |
| `timestamp_utc` | ISO-8601 | Audit chronology |

Escalations additionally carry an `escalation_type` drawn from a closed
enumeration (`conflicting_evidence`, `no_evidence`, `unverified_evidence`,
`insufficient_history`, `missing_required_evidence`, `tool_failure`,
`guardrail_violation`, `human_rejected`, `human_no_decision`). An audit consumer
distinguishes "a planner refused this" from "the data was untrustworthy" by
field, never by parsing prose.

Human decisions carry `planner_id`. An unattributed approval is an evaluation
failure, not a warning.

## Evidence integrity

Contradiction is defined *within* a purchase order, between updates sharing a
reported date. This matters more than it sounds: comparing statuses across rows
means any SKU with two open POs looks contradictory, and a system that escalates
the normal operating case gets switched off within a week. Equally, a later
verified update correcting an earlier one is a correction, not a conflict.

The evaluation suite pins both directions — `multi_po_clean` and
`superseded_update` must **not** escalate; `conflicting_evidence` and
`no_evidence` must.

## Bounded actions

`policy.select_action` bounds every transfer by three independent limits: the
computed safety-stock gap, the real transferable surplus at the source site
(on-hand above *its own* safety stock, so closing one shortage cannot open
another), and a named `MAX_TRANSFER_UNITS` single-proposal cap. Any residual
exposure is stated explicitly and handed to the planner rather than quietly
absorbed. Every action carries its own rollback instruction.

## Failure controls and observability

The workflow fails closed on absent, unverified, or contradictory shipment data,
on insufficient demand history, on tool failure, and on guardrail violation. Tool
failures record `executed_by: local_duckdb` and are never attributed to the model.

Production monitoring should track data freshness, tool failure rate, guardrail
violation rate by violation code, escalation rate split by `escalation_type`,
planner approval and override rates, narrative grounding failures, latency per
node, and model cost per triage. Deploy versioned graphs and policies through
replay tests and canaries; replace the demo's `MemorySaver` with durable
checkpoint storage so workflow changes can be rolled back without downtime.
