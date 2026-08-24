# Supply-chain disruption triage agents

An auditable, safety-first **Track A — Agentic / Multi-Agent AI** submission
for Avathon's S1 Supply Chain Risk & Optimization scenario. The system detects
supplier-delay evidence, estimates inventory impact, proposes a reversible
response, and pauses for mandatory human approval before simulated execution.

## Problem and approach

Supplier delays, volatile demand, and imbalanced stock create a connected
decision problem: detecting a late shipment alone is insufficient unless the
system quantifies the likely stockout and proposes a policy-compliant response.
This repository implements three genuinely separate LangGraph nodes:

1. **Risk Detection & Evidence Agent** retrieves and validates delay/inventory
   evidence through a read-only DuckDB tool.
2. **Demand & Impact Agent** independently retrieves demand history and
   computes an interpretable short-horizon exposure estimate.
3. **Replenishment Recommendation Agent** produces a bounded and reversible
   proposal. It cannot make ERP or purchase-order changes.

LangGraph provides an explicit state graph, conditional routing, checkpointing,
and a mandatory human-in-the-loop interrupt. See
[the architecture decision record](docs/architecture.md) for framework
alternatives, message schema, safety controls, and production considerations.

This is agentic decision orchestration, not a claim that an LLM should replace
forecasting or policy math. Tool results and decisions are structured and
reproducible; a future LLM integration can create planner-facing explanations
without receiving authority to alter tool calls or operational actions.

## Data source

No data was supplied with the assessment. This project uses seeded synthetic
data because the required joined grain—daily demand, inventory, POs, and
shipment-status updates—rarely exists in a public dataset. The generator
creates demand seasonality and dispersion, then injects a known five-day
supplier delay for a critical SKU. A separate scenario adds contradictory,
unverified shipment updates to test fail-closed behavior.

See [data assumptions and limitations](docs/data.md). The generation script,
seed, distributions, and injected events are all documented and reproducible.

## Repository layout

```text
data/generated/       Generated runtime CSVs (not committed)
docs/                 Data and architecture decisions
results/              Generated JSONL agent traces (not committed)
src/avathon/          Agents, tools, LangGraph orchestration, CLI
tests/                Success and failure-path tests
write-up/             Reserved for the 1–2 page PDF write-up
```

## Setup

Python **3.12** is required and dependencies are pinned in
[`requirements.txt`](requirements.txt).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

## Reproduce the complete demo

Run the two required end-to-end paths from the repository root:

```bash
python -m avathon.run_demo --scenario success --project-root .
python -m avathon.run_demo --scenario failure --project-root .
pytest -q
```

The commands generate:

- `results/trace_success.jsonl`: verified supplier delay → risk/impact
  analysis → mandatory human approval → simulated approved action.
- `results/trace_failure.jsonl`: contradictory/unverified shipment evidence →
  escalation, with no recommendation or action proposal.

Inspect a trace with:

```bash
cat results/trace_success.jsonl
```

## Interactive Streamlit console

The repository also includes a visual test console for the two end-to-end
workflow cases, the complete typed trace, information flow between agents, and
the mandatory human approval step. From the repository root:

```bash
streamlit run app.py
```

Choose **Verified delay: human approval required** to review a recommendation
and approve or reject the simulated action. Choose **Conflicting shipment data:
fail closed** to confirm that the workflow escalates without proposing an
action. The interface can download the current trace as JSONL.

## Safety boundaries

- The DuckDB tooling is read-only from the agent perspective and exposes only
  allow-listed, parameterized operations.
- Contradictory or unverified evidence causes escalation rather than guesswork.
- The graph stops before human review; no automated ERP/WMS action exists.
- Recommended actions include evidence IDs, assumptions, confidence, and a
  rollback instruction.

## Current limitations

The synthetic data validates workflow behavior and traceability, not causal
performance on a live network. A production pilot would first replay historical
ERP outcomes in shadow mode, calibrate threshold policies with planners, and
replace the in-memory checkpoint store with durable infrastructure.
