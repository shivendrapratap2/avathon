"""Interactive observability console for the supply-chain agent workflow.

Run with: streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from uuid import uuid4

import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from avathon.data_generation import generate  # noqa: E402
from avathon.workflow import build_workflow  # noqa: E402


SCENARIOS = {
    "Verified delay: human approval required": {
        "data_scenario": "success",
        "scenario_id": "verified_delay",
        "test_name": "test_verified_delay_requires_human_review_then_allows_simulated_action",
        "description": (
            "A verified five-day delay for SKU-CRITICAL at Pune-DC should create a "
            "bounded recommendation and pause for a human decision."
        ),
    },
    "Conflicting shipment data: fail closed": {
        "data_scenario": "conflicting_evidence",
        "scenario_id": "conflicting_shipment_data",
        "test_name": "test_conflicting_shipment_data_fails_closed_without_recommendation",
        "description": (
            "Contradictory, unverified shipment updates must escalate to a planner "
            "without creating a recommendation."
        ),
    },
}


def state() -> dict:
    """Return the latest persisted LangGraph state for the current browser session."""
    run = st.session_state.get("run")
    return run["graph"].get_state(run["config"]).values if run else {}


def message_rows(messages: list[dict]) -> list[dict]:
    """Make the typed agent handoffs easy to scan without hiding raw detail."""
    return [
        {
            "#": index,
            "From": message["sender"],
            "To": message["recipient"],
            "Message": message["message_type"],
            "Confidence": f"{message['confidence']:.0%}",
            "Evidence": ", ".join(message["evidence_refs"]) or "-",
        }
        for index, message in enumerate(messages, start=1)
    ]


def start_scenario(selection: str, api_key: str | None = None, model: str = "gpt-5.6") -> None:
    """Generate the selected scenario and run until completion or HITL interrupt."""
    scenario = SCENARIOS[selection]
    data_dir = ROOT / "data" / "generated"
    generate(data_dir, scenario=scenario["data_scenario"])
    graph = build_workflow(data_dir, api_key=api_key, model=model)
    config = {"configurable": {"thread_id": f"ui-{uuid4().hex}"}}
    initial = {
        "trace_id": f"ui-{uuid4().hex[:10]}",
        "scenario_id": scenario["scenario_id"],
        "sku": "SKU-CRITICAL",
        "site": "Pune-DC",
        "decision": "pending",
        "messages": [],
    }
    graph.invoke(initial, config=config)
    st.session_state.run = {
        "graph": graph, "config": config, "selection": selection,
        "llm_enabled": bool(api_key), "model": model if api_key else None,
    }


def decide(decision: str) -> None:
    """Resume only the paused success path with an explicit human decision."""
    run = st.session_state.run
    run["graph"].update_state(run["config"], {"decision": decision})
    run["graph"].invoke(None, config=run["config"])


def trace_jsonl(messages: list[dict]) -> str:
    return "".join(json.dumps(message, default=str) + "\n" for message in messages)


st.set_page_config(page_title="Supply-chain agent console", page_icon="⛓", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background: #f6f8fb; }
      .block-container { max-width: 1200px; padding-top: 2rem; }
      .agent-card { background: white; border: 1px solid #dce4ef; border-radius: 12px;
                    padding: 1rem; min-height: 145px; }
      .agent-label { color: #166a7a; font-weight: 700; font-size: 0.9rem; }
      .trace-arrow { color: #6b7280; font-size: 1.1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Supply-chain agent observability console")
st.caption("A safe simulation: agents analyze and recommend; a human authorizes every operational step.")

with st.sidebar:
    st.header("Scenario control")
    selection = st.radio("Choose a workflow test", list(SCENARIOS), label_visibility="collapsed")
    chosen = SCENARIOS[selection]
    st.caption(chosen["test_name"])
    st.info(chosen["description"])
    st.divider()
    st.subheader("LLM tool-calling test")
    use_llm = st.toggle("Use GPT for tool selection", value=False)
    api_key = ""
    model = "gpt-5.6"
    if use_llm:
        api_key = st.text_input(
            "OpenAI API key", type="password",
            help="Used only in this server session; never written to disk or the trace.",
        )
        model = st.text_input(
            "Model", value=model, help="Use a model ID enabled for your OpenAI API project."
        )
        st.caption("The model may select read-only tools only. Invalid tool calls stop the workflow safely.")
    if st.button("Run selected scenario", type="primary", width="stretch"):
        if use_llm and not api_key:
            st.error("Enter an OpenAI API key to enable GPT tool calling.")
        else:
            try:
                start_scenario(selection, api_key=api_key or None, model=model)
            except RuntimeError as error:
                st.error(str(error))
    if st.button("Clear current run", width="stretch"):
        st.session_state.pop("run", None)
        st.rerun()

st.subheader("Agent responsibilities")
agent_cols = st.columns(3)
agent_cards = [
    ("1. Risk Detection & Evidence", "Queries read-only shipment, PO, and inventory facts. Stops on contradictory evidence."),
    ("2. Demand & Impact", "Retrieves demand history and calculates days of cover with a transparent uncertainty signal."),
    ("3. Replenishment Recommendation", "Creates a policy-bounded, reversible proposal. It cannot execute an ERP change."),
]
for column, (label, description) in zip(agent_cols, agent_cards):
    with column:
        st.markdown(f'<div class="agent-card"><div class="agent-label">{label}</div><br>{description}</div>', unsafe_allow_html=True)

if "run" not in st.session_state:
    st.info("Choose a scenario in the left panel, then select Run selected scenario.")
    st.stop()

current = state()
messages = current.get("messages", [])
message_types = {message["message_type"] for message in messages}
proposal = current.get("proposal")
awaiting_human = proposal is not None and "human_decision" not in message_types
has_escalation = "escalation" in message_types
is_approved = messages and messages[-1]["message_type"] == "action_proposal"

if st.session_state.run["llm_enabled"]:
    st.success(
        f"GPT tool selection enabled with `{st.session_state.run['model']}`. "
        "Tool call IDs and response IDs are included in the trace."
    )
else:
    st.info("Deterministic tool mode: select GPT tool calling in the sidebar to test model-selected, validated tool calls.")

st.divider()
metric_cols = st.columns(4)
metric_cols[0].metric("Trace events", len(messages))
metric_cols[1].metric("Agent nodes reached", len({m["sender"] for m in messages if m["sender"].endswith("agent")}))
metric_cols[2].metric("Evidence references", len({ref for m in messages for ref in m["evidence_refs"]}))
metric_cols[3].metric("Workflow status", "Awaiting approval" if awaiting_human else "Escalated" if has_escalation else "Approved" if is_approved else "Running")

if awaiting_human:
    st.warning("Human checkpoint reached. The recommendation is blocked until a planner decides.")
    st.subheader("Planner decision")
    st.json(proposal, expanded=False)
    approve, reject = st.columns(2)
    with approve:
        if st.button("Approve simulated action", type="primary", width="stretch"):
            decide("approve")
            st.rerun()
    with reject:
        if st.button("Reject and escalate", width="stretch"):
            decide("reject")
            st.rerun()
elif has_escalation:
    reason = next(
        (message["payload"].get("reason") for message in reversed(messages) if message["message_type"] == "escalation"),
        "Workflow stopped safely.",
    )
    st.error(f"Workflow escalated safely: {reason}")
elif is_approved:
    st.success("Human approval recorded. The system emitted a simulated action and audit event.")

st.divider()
st.subheader("Trace and information movement")
st.caption("Each row is a typed message. Expand an event to inspect the exact payload, assumptions, and evidence IDs.")
st.dataframe(message_rows(messages), width="stretch", hide_index=True)

for index, message in enumerate(messages, start=1):
    title = f"{index:02d}  {message['sender']}  ->  {message['recipient']}  |  {message['message_type']}"
    with st.expander(title):
        top = st.columns(2)
        top[0].write(f"**Confidence:** {message['confidence']:.0%}")
        top[1].write(f"**Evidence:** {', '.join(message['evidence_refs']) or 'None'}")
        st.write("**Payload**")
        st.json(message["payload"], expanded=True)
        if message["assumptions"]:
            st.write("**Assumptions**")
            for assumption in message["assumptions"]:
                st.write(f"- {assumption}")

st.download_button(
    "Download this trace as JSONL",
    data=trace_jsonl(messages),
    file_name=f"{current['trace_id']}.jsonl",
    mime="application/x-ndjson",
)

with st.expander("Safety boundary"):
    st.write(
        "The app never writes to an ERP, WMS, purchase-order system, or inventory system. "
        "It demonstrates evidence retrieval, governed recommendation, a human decision, and an audit trace."
    )
