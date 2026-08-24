"""Interactive console for the GPT-planned supply-chain triage workflow.

Run with: streamlit run app.py

The console exists to make three things visible that a JSONL trace makes hard to
see: what the model actually chose to do, where the boundary stopped it, and the
fact that no operational action exists without a named human approving it.
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

from avathon.data_generation import SCENARIOS, generate  # noqa: E402
from avathon.display import proposal_rows, trace_rows  # noqa: E402
from avathon.evaluation import primary_expectations  # noqa: E402
from avathon.llm import DEFAULT_MODEL, LLMPlanner, rogue_planner_factory  # noqa: E402
from avathon.policy import MAX_TRANSFER_UNITS  # noqa: E402
from avathon.workflow import build_workflow  # noqa: E402


SKU = "SKU-CRITICAL"
SITE = "Pune-DC"

#: Scenario labels in the order a reviewer should walk them: the working path
#: first, then the two ways the system is required to stop, then the two cases
#: that must *not* stop it.
SCENARIO_ORDER = [
    ("verified_delay", "Verified delay", "success"),
    ("healthy_stock", "Healthy cover", "success"),
    ("multi_po_clean", "Several open POs", "regression"),
    ("superseded_update", "Corrected update", "regression"),
    ("conflicting_evidence", "Contradictory updates", "failure"),
    ("no_evidence", "No purchase order", "failure"),
]

EXPECTED = primary_expectations()

AGENTS = [
    ("risk_agent", "1 · Risk detection & evidence",
     "Plans its own queries to establish whether a delay is real and whether the "
     "shipment evidence is internally consistent."),
    ("impact_agent", "2 · Demand & impact",
     "Chooses a lookback window and retrieves demand independently. Exposure is "
     "computed against safety stock, not against bare on-hand."),
    ("replenishment_agent", "3 · Replenishment recommendation",
     "Must ground any transfer in real surplus at another site. It cannot execute "
     "an ERP change."),
]

STATUS_STYLE = {
    "pending": ("#94a3b8", "Not reached"),
    "running": ("#0891b2", "Working"),
    "done": ("#059669", "Complete"),
    "stopped": ("#dc2626", "Stopped here"),
}


# -- page ---------------------------------------------------------------------

st.set_page_config(
    page_title="Supply-chain agent console", page_icon="⛓", layout="wide"
)
st.markdown(
    """
    <style>
      .stApp { background: #f6f8fb; }
      .block-container { max-width: 1280px; padding-top: 2rem; }
      .agent-card { background: #fff; border: 1px solid #dce4ef; border-radius: 12px;
                    padding: 0.9rem 1rem; min-height: 172px; }
      .agent-label { color: #0f4c5c; font-weight: 700; font-size: 0.92rem; }
      .agent-body { color: #475569; font-size: 0.82rem; line-height: 1.45; margin-top: .5rem; }
      .pill { display:inline-block; padding: 2px 10px; border-radius: 999px;
              font-size: 0.72rem; font-weight: 700; color: #fff; }
      .banner { background:#0f172a; color:#e2e8f0; padding:.7rem 1rem;
                border-radius:10px; font-size:.83rem; }
      .verdict-ok { color:#059669; font-weight:600; }
      .verdict-block { color:#dc2626; font-weight:600; }
      code { font-size: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Supply-chain disruption triage")
st.caption(
    "GPT plans the investigation. A deterministic policy engine computes every number. "
    "A named human authorizes every operational step."
)


# -- helpers ------------------------------------------------------------------


def current_state() -> dict:
    run = st.session_state.get("run")
    return run["graph"].get_state(run["config"]).values if run else {}


def is_paused() -> bool:
    run = st.session_state.get("run")
    return bool(run and run["graph"].get_state(run["config"]).next)


def start(scenario: str, api_key: str, model: str, planner_id: str, rogue: bool) -> None:
    data_dir = ROOT / "data" / "generated" / scenario
    generate(data_dir, scenario=scenario)

    if rogue:
        graph = build_workflow(data_dir, planner_factory=rogue_planner_factory(SKU, SITE))
        model_label = "scripted adversary"
    else:
        planner = LLMPlanner(api_key=api_key, model=model)
        graph = build_workflow(data_dir, planner=planner)
        model_label = model

    config = {"configurable": {"thread_id": f"ui-{uuid4().hex}"}}
    graph.invoke(
        {
            "trace_id": f"ui-{uuid4().hex[:10]}", "scenario_id": scenario,
            "sku": SKU, "site": SITE, "planner_id": planner_id,
            "decision": "pending", "messages": [],
        },
        config=config,
    )
    st.session_state.run = {
        "graph": graph, "config": config, "scenario": scenario,
        "model": model_label, "rogue": rogue, "planner_id": planner_id,
    }


def decide(decision: str) -> None:
    run = st.session_state.run
    run["graph"].update_state(run["config"], {"decision": decision})
    run["graph"].invoke(None, config=run["config"])


def agent_status(messages: list[dict], agent: str, stopped_at: str | None) -> str:
    if stopped_at == agent:
        return "stopped"
    if any(message["sender"] == agent for message in messages):
        return "done"
    return "pending"


def pill(status: str) -> str:
    colour, label = STATUS_STYLE[status]
    return f'<span class="pill" style="background:{colour}">{label}</span>'


# -- sidebar ------------------------------------------------------------------

with st.sidebar:
    st.header("Run a scenario")

    api_key = st.text_input(
        "OpenAI API key", type="password",
        help="Held in this browser session only. Never written to disk, to "
             "`.streamlit/secrets.toml`, or into the trace.",
    )
    model = st.text_input("Model", value=DEFAULT_MODEL)
    planner_id = st.text_input(
        "Approving planner", value="planner@example.com",
        help="Recorded on the approval event. An unattributed approval fails evaluation.",
    )

    st.divider()
    labels = {key: f"{label}  ·  {kind}" for key, label, kind in SCENARIO_ORDER}
    scenario = st.radio(
        "Scenario", [key for key, _, _ in SCENARIO_ORDER],
        format_func=lambda key: labels[key],
    )
    st.caption(SCENARIOS[scenario])

    expectation = EXPECTED.get(scenario)
    if expectation:
        target = expectation.expect_escalation or f"proposal · {expectation.expect_action}"
        st.info(f"**Required outcome:** {target}\n\n{expectation.rationale}")

    run_clicked = st.button("Run scenario", type="primary", width="stretch")

    st.divider()
    st.subheader("Guardrail probe")
    st.caption(
        "Replaces the risk agent's planner with one that deliberately queries a "
        "different SKU and site. A live model cannot be relied on to misbehave on "
        "cue, so the boundary is tested with a known offender."
    )
    probe_clicked = st.button("Run adversarial planner", width="stretch")

    if st.button("Clear run", width="stretch"):
        st.session_state.pop("run", None)
        st.rerun()

if run_clicked or probe_clicked:
    if not probe_clicked and not api_key:
        st.sidebar.error("An OpenAI API key is required: the planner is the model.")
    else:
        label = "Adversarial planner" if probe_clicked else f"{model} planning the investigation"
        with st.spinner(f"{label}…"):
            try:
                start(scenario, api_key, model, planner_id or "unattributed", probe_clicked)
            except Exception as error:  # surfaced, never swallowed
                st.session_state.pop("run", None)
                st.error(f"{type(error).__name__}: {error}")


# -- architecture strip -------------------------------------------------------

state = current_state()
messages = state.get("messages", [])
types = [message["message_type"] for message in messages]
escalation_type = state.get("escalation_type")
stopped_at = None
if escalation_type:
    stopped_at = next(
        (
            message["sender"] for message in reversed(messages)
            if message["message_type"] in ("escalation", "guardrail_block")
            and message["sender"].endswith("agent")
        ),
        None,
    )

st.markdown(
    '<div class="banner">The model selects read-only tools and explains findings. '
    "It never produces a quantity, a risk level, or an execution permission — those "
    "come from the policy engine, and the trace records which is which.</div>",
    unsafe_allow_html=True,
)
st.write("")

columns = st.columns(3)
for column, (agent, title, description) in zip(columns, AGENTS):
    status = agent_status(messages, agent, stopped_at) if messages else "pending"
    with column:
        st.markdown(
            f'<div class="agent-card"><div class="agent-label">{title}</div>'
            f'<div style="margin-top:.45rem">{pill(status)}</div>'
            f'<div class="agent-body">{description}</div></div>',
            unsafe_allow_html=True,
        )

if "run" not in st.session_state:
    st.info(
        "Enter an API key, choose a scenario, and run it. The guardrail probe needs "
        "no key — it uses a scripted adversary."
    )
    st.stop()

run = st.session_state.run
proposal = state.get("proposal")
awaiting = is_paused()
approved = "action_proposal" in types

st.divider()

# -- outcome ------------------------------------------------------------------

expectation = EXPECTED.get(run["scenario"])
if expectation and not run["rogue"]:
    observed = escalation_type or (f"proposal:{proposal['action']}" if proposal else "incomplete")
    required = expectation.expect_escalation or f"proposal:{expectation.expect_action}"
    if awaiting:
        pass  # not yet resolvable; the human gate is still open
    elif observed == required:
        st.success(f"Required outcome met — `{observed}`")
    else:
        st.error(f"Outcome mismatch — required `{required}`, observed `{observed}`")

metrics = st.columns(5)
metrics[0].metric("Trace events", len(messages))
metrics[1].metric("Tool calls allowed", sum(
    message["payload"].get("tool_calls", 0)
    for message in messages if message["message_type"] == "planner_step"
))
metrics[2].metric("Evidence refs", len({
    ref for message in messages for ref in message["evidence_refs"]
}))
metrics[3].metric("Guardrail blocks", types.count("guardrail_block"))
metrics[4].metric(
    "Status",
    "Awaiting approval" if awaiting else "Escalated" if escalation_type
    else "Approved" if approved else "Complete",
)

# -- human gate ---------------------------------------------------------------

if awaiting and proposal:
    st.warning(
        f"**Human checkpoint.** Nothing executes until `{run['planner_id']}` decides. "
        "The proposal below is recorded as blocked."
    )
    left, right = st.columns([3, 2])
    with left:
        st.markdown("##### Proposed action")
        st.table(proposal_rows(proposal))
        st.markdown(f"**Why:** {proposal['reason']}")
        for constraint in proposal.get("constraints", []):
            st.markdown(f"- ⚠️ {constraint}")
        st.markdown(f"**Rollback:** {proposal['rollback']}")
    with right:
        approval_message = next(
            (m for m in messages if m["message_type"] == "approval_request"), None
        )
        if approval_message and approval_message["narrative"]:
            st.markdown("##### Model explanation")
            st.info(approval_message["narrative"])
            st.caption(
                "Every number above was verified to appear in the computed facts before "
                "this text was allowed into the trace."
            )
        st.markdown("##### Decision")
        if st.button("Approve simulated action", type="primary", width="stretch"):
            decide("approve")
            st.rerun()
        if st.button("Reject and escalate", width="stretch"):
            decide("reject")
            st.rerun()

elif escalation_type:
    reason = next(
        (
            message["payload"].get("reason") for message in reversed(messages)
            if message["message_type"] == "escalation"
        ),
        "Workflow stopped safely.",
    )
    st.error(f"**Stopped — `{escalation_type}`**\n\n{reason}")
    if types.count("approval_request") == 0:
        st.caption("No recommendation was produced and no action was proposed.")

elif approved:
    final = messages[-1]["payload"]
    st.success(
        f"Approved by `{final['approved_by']}` — simulated "
        f"{final['action']} of {final['quantity']} units recorded to the audit log."
    )

# -- planner activity ---------------------------------------------------------

st.divider()
st.subheader("What the planner did")
st.caption(
    "Each tool call was checked against the agent's allow-list, argument schema, "
    "value bounds, and investigation scope before the query ran."
)

planner_steps = [m for m in messages if m["message_type"] == "planner_step"]
blocks = [m for m in messages if m["message_type"] == "guardrail_block"]

for message in planner_steps:
    payload = message["payload"]
    header = (
        f"{payload['model']} · {message['sender']} · "
        f"{payload['tool_calls']} tool call(s) · {', '.join(payload['tools_selected']) or 'none'}"
    )
    with st.expander(header, expanded=len(planner_steps) == 1):
        for step in payload["steps"]:
            if step["kind"] == "tool_call":
                st.markdown(
                    f"↳ **calls** `{step['tool_name']}`  "
                    f"`{json.dumps(step.get('arguments', {}))}`"
                )
            elif step["kind"] == "tool_result":
                st.markdown(
                    f"&nbsp;&nbsp;&nbsp;<span class='verdict-ok'>allowed</span> · "
                    f"{step.get('row_count', 0)} rows · `{step.get('query_id','')}`",
                    unsafe_allow_html=True,
                )
            elif step["kind"] == "narrative" and step.get("text"):
                st.caption(f"planner: {step['text']}")
        if message["narrative"]:
            st.caption(f"planner: {message['narrative']}")

for message in blocks:
    payload = message["payload"]
    st.error(
        f"**Guardrail block — `{payload['violation']}`**\n\n"
        f"`{payload['selected_by']}` proposed `{payload['tool_name']}`. "
        f"{payload['detail']}\n\nThe query never reached the database "
        f"(`executed: {payload['executed']}`)."
    )

# -- trace --------------------------------------------------------------------

st.divider()
st.subheader("Trace")
st.caption("Every typed message exchanged. Expand any event for its exact payload.")

st.dataframe(trace_rows(messages), width="stretch", hide_index=True)

for index, message in enumerate(messages, start=1):
    title = (
        f"{index:02d}  {message['sender']} → {message['recipient']}  |  "
        f"{message['message_type']}"
    )
    with st.expander(title):
        head = st.columns(2)
        head[0].write(f"**Confidence:** {message['confidence']:.0%}")
        head[1].write(f"**Evidence:** {', '.join(message['evidence_refs']) or 'None'}")
        if message["narrative"]:
            st.info(message["narrative"])
        grounding = message["payload"].get("narrative_grounding")
        if grounding and not grounding.get("ok"):
            st.warning(
                "Model explanation suppressed — it contained numbers absent from the "
                f"computed facts: {grounding.get('ungrounded_numbers', grounding)}"
            )
        st.json(message["payload"], expanded=False)
        for assumption in message["assumptions"]:
            st.markdown(f"- _{assumption}_")

st.download_button(
    "Download trace as JSONL",
    data="".join(json.dumps(m, default=str) + "\n" for m in messages),
    file_name=f"{state['trace_id']}.jsonl",
    mime="application/x-ndjson",
)

with st.expander("Safety boundary"):
    st.markdown(
        f"""
- The model chooses **which** read-only tools to call, in what order, and with what
  lookback. It never writes SQL and never reaches a tool that has not passed the
  per-agent allow-list, argument schema, value bounds, and investigation-scope check.
- Every figure shown — days of cover, safety-stock gap, quantity, risk level — is
  computed deterministically from tool results. Model text is explanatory and is
  checked for numeric grounding before it enters the trace.
- No proposal may exceed real transferable surplus at another site, or the
  {MAX_TRANSFER_UNITS}-unit single-proposal policy cap.
- There is no ERP, WMS, or purchase-order write path in this system at all.
  Approval records a simulated action to an audit log and nothing more.
"""
    )
