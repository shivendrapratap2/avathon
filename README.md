# Supply-chain disruption triage agents

An auditable, safety-first **Track A — Agentic / Multi-Agent AI** submission for
Avathon's S1 Supply Chain Risk & Optimization scenario. A GPT planner drives three
separately accountable LangGraph agents through a bounded investigation; a
deterministic policy engine turns the retrieved evidence into figures; and the
workflow pauses for a named human before anything is recorded as an action.

## The design decision this project is actually about

Two failure modes bracket this problem. Hand the whole thing to an LLM and it
will confidently invent a purchase order. Hardcode the whole thing and you have a
rules engine wearing an agent costume. This system splits the two:

| Concern | Owner |
| --- | --- |
| Which evidence to gather, in what order, with what lookback | **GPT planner** |
| Whether a proposed tool call may run at all | **Guardrail layer** |
| Every number — cover, gap, quantity, risk level | **Policy engine** |
| Whether anything happens | **Human approval** |
| Explaining the result to a planner | **GPT narrator**, numerically grounded |

The model can be wrong about *how to investigate* — recoverable and visible in
the trace. It cannot be wrong about *what to do*.

## The three agents

1. **Risk Detection & Evidence** plans its own queries to establish whether a
   delay is real and whether the shipment evidence is internally consistent and
   verified.
2. **Demand & Impact** independently retrieves demand, choosing its own lookback
   within an approved window, and computes exposure against safety stock — the
   floor to protect, not buffer to consume.
3. **Replenishment Recommendation** must ground any transfer in real surplus at
   another site before proposing it. It has no ERP or purchase-order write path.

Each agent's tool allow-list is deliberately wider than its minimum requirement,
so tool selection is a genuine decision rather than a single legal move. See the
[architecture decision record](docs/architecture.md).

## Setup

Python 3.11+ (3.12 recommended).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

## The console

**In a hurry? [DEMO.md](DEMO.md) is a five-minute click-through** covering the
working path, the fail-closed path, and the guardrail boundary.

```bash
streamlit run app.py
```

Enter an OpenAI API key in the sidebar — the planner *is* the model, and there is
no hidden deterministic fallback. The key is held for the browser session only and
is never written to disk, to `.streamlit/secrets.toml`, or into the trace.

Six scenarios, in the order worth walking them:

| Scenario | What it demonstrates |
| --- | --- |
| **Verified delay** | The working path: investigation → bounded proposal → human gate |
| **Healthy cover** | Restraint: ample stock produces `monitor`, not a transfer |
| **Several open POs** | Normal complexity must **not** escalate |
| **Corrected update** | A revised ETA is a correction, not a contradiction |
| **Contradictory updates** | Fail closed: escalate, no recommendation |
| **No purchase order** | Fail closed on absent evidence |

The console shows what the model actually chose to do at each step, the
validation verdict on every tool call, and the full typed trace.

**Guardrail probe** needs no API key. It swaps the risk agent's planner for one
that deliberately queries a different SKU and site, and shows the call being
blocked before it reaches the database. A live model cannot be relied on to
misbehave on cue, so the boundary is tested with a known offender.

## Reproduce from the command line

```bash
# Reference plan — deterministic, no API key, no spend
python -m avathon.run_demo --scenario verified_delay
python -m avathon.run_demo --scenario conflicting_evidence
python -m avathon.evaluation
pytest

# Live GPT planning
export OPENAI_API_KEY=sk-...
python -m avathon.run_demo --scenario verified_delay --live
python -m avathon.evaluation --live
```

Traces land in `results/trace_<scenario>.jsonl`; the evaluation writes
`results/eval_summary.md` and `.json`.

## Evaluation

Eight cases, balanced between "must stop" and "must proceed", checked against
per-case outcomes and universal invariants. Two metrics carry the weight:

- **Missed escalations** — proceeded when it should have stopped. The only class
  that causes direct operational harm. Target 0.
- **False escalations** — stopped on a normal operating case. The mechanism by
  which an assistive system gets switched off. Target 0.

Cases run against both a scripted reference plan and a live model, so a
regression can be attributed to the graph or to the model. Full method and the
case matrix: [docs/evaluation.md](docs/evaluation.md).

## Safety boundaries

- Three named, parameterized, read-only DuckDB operations. Agents never supply
  SQL, and a hallucinated call becomes a rejected call, never an unbounded query.
- Every model-proposed call is checked against the agent's allow-list, argument
  schema, value bounds, investigation scope, and a duplicate-call guard **before**
  execution. A violation escalates; it never degrades into a recommendation.
- Contradiction is detected *within* a purchase order, between updates sharing a
  reported date — so several open POs and corrected ETAs stay routine.
- Exposure is measured against safety stock, and every transfer is bounded by the
  computed gap, the real surplus at the source site, and a named policy cap.
  Residual exposure is stated, not absorbed.
- Model-written explanations are checked for numeric grounding; an invented figure
  suppresses the narrative and is recorded in the trace.
- Escalations carry a machine-readable `escalation_type`; approvals carry a
  `planner_id`. An unattributed approval fails evaluation.
- Conversation state is held locally with `store=False`, so no operational
  evidence is retained provider-side.
- There is no ERP, WMS, or purchase-order write path anywhere in the system.

## Data

No data was supplied with the assessment. The required joined grain — daily
demand, inventory, open POs, and successive shipment updates *carrying
provenance*, at a common SKU/site key — does not exist publicly. The generator is
seeded and every scenario is one controlled perturbation of the same base. See
[data assumptions and limitations](docs/data.md).

## Repository layout

```text
app.py                Streamlit console
data/generated/       Generated runtime CSVs (not committed)
docs/                 Architecture, data, and evaluation decisions
results/              Generated traces and evaluation reports (not committed)
src/avathon/
  agents.py           Three agents: planner + policy composition
  data_generation.py  Seeded scenarios
  evaluation.py       Scenario-matrix safety evaluation
  guardrails.py       The enforcement boundary
  llm.py              Planner loop, narrator, grounding check
  policy.py           Deterministic evidence integrity, impact, action bounds
  schemas.py          Typed message contract
  tools.py            Read-only DuckDB operations
  workflow.py         LangGraph orchestration
tests/                Policy, guardrail, planner, workflow, evaluation, UI
write-up/             Track A technical write-up, and the script that builds it
```

The write-up PDF is regenerated with `python write-up/build_writeup.py`, so the
document stays reproducible alongside the code it describes.

## Current limitations

The synthetic data validates workflow behaviour and traceability, not causal
performance on a live network, and this evaluation makes no forecast-accuracy
claim. The checkpoint store is in-memory. A production pilot would replay
historical ERP outcomes in shadow mode with planner adjudication, calibrate
threshold policies against real demand, and replace the checkpoint store with
durable infrastructure before widening the scope of recommendations.
