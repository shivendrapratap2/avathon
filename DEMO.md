# Five-minute demo

A reviewer-facing walkthrough. Three clicks show the working path, the
fail-closed path, and the boundary. Everything after that is optional.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Enter an OpenAI API key in the sidebar. The planner *is* the model — there is no
deterministic fallback hiding behind it. The **guardrail probe** at the bottom of
the sidebar needs no key, and so does the whole evaluation suite:

```bash
python -m avathon.evaluation      # 8-case safety matrix, no key, no spend
```

---

## 1 · Verified delay — the working path

**Sidebar → "Verified delay · success" → Run scenario.** (~2 min)

Three planner blocks appear under *What the planner did*, one per agent. Each
agent was given an objective and a tool allow-list wider than its minimum
requirement, then chose its own calls.

| Stage | What you should see |
| --- | --- |
| Risk | `PO-001` from `SUP-A`, **5 days** late, revised ETA `2026-08-26`. On hand **175** against safety stock **500** |
| Impact | ~**92.9** units/day, **1.9** days of cover, **464** units projected over the delay, safety-stock gap **789** units |
| Proposal | `transfer_stock` **400** units from **Mumbai-DC** — held at the human gate |

**The number worth pausing on is 400, not 789.** The quantity is the minimum of
three independent limits, all visible in the proposal panel:

- the computed gap — 789 units
- **real transferable surplus at Mumbai-DC — 400 units**, its on-hand above *its
  own* safety stock, so closing one shortage cannot open another
- the named single-proposal policy cap — 600 units

The binding constraint is stated explicitly, and so is what it leaves behind:
*"389 units of exposure remain after this transfer and need a planner decision."*
Residual exposure is handed back, not quietly absorbed.

Two more things on this screen:

- The proposal is recorded `blocked_pending_human_approval` at the moment it is
  created, before anyone sees it.
- The **model explanation** beside the decision buttons was written by GPT, and
  every numeric token in it was checked against the computed facts before it was
  allowed into the trace. An invented figure suppresses the whole explanation and
  is recorded in the trace — the decision is unaffected, because the decision
  never read the text.

**Click Approve.** The final event carries `approved_by`, and cites all three
query IDs — the risk, demand, and alternate-site queries that produced it.

---

## 2 · Contradictory updates — fail closed

**Sidebar → "Contradictory updates · failure" → Run scenario.** (~1 min)

Two shipment updates for `PO-001` share a reported date and disagree; one is
unverified.

The workflow stops at agent 1. Four events total, and the two that matter are the
ones that are **absent**: no `approval_request`, no `action_proposal`. No
recommendation was produced at all — the system does not guess an ETA and hedge.

`escalation_type: conflicting_evidence` is a field drawn from a closed
enumeration, not prose. An audit consumer distinguishes "the data was
untrustworthy" from "a planner refused this" (`human_rejected`) without parsing
a sentence. Try **Reject and escalate** on scenario 1 to see the other one.

---

## 3 · Guardrail probe — the boundary

**Sidebar → Run adversarial planner.** (~1 min, no API key)

This swaps the risk agent's planner for one that deliberately queries a different
SKU and site. A live model cannot be relied on to misbehave on cue, so the
boundary is tested with a known offender.

Result: `out_of_scope_sku`, **Tool calls allowed: 0**, and `executed: false`. The
query never reached the database. Validation happens before execution, not after,
and a violation escalates rather than degrading into a recommendation.

---

## If you have another three minutes

### The regression pair — the point of the whole project

**"Several open POs · regression"** and **"Corrected update · regression"** must
both **proceed**, not escalate. The first shows `open_po_count: 2` with
`delayed_po_count: 1` and still reaches a proposal.

My first version detected contradictions by comparing statuses across returned
rows. That meant any SKU with two open purchase orders looked contradictory. It
passed its tests, because the fixture contained exactly one PO per SKU. In
production it would have escalated nearly everything and been switched off inside
a week.

Contradiction is now defined *within* a purchase order, between updates sharing a
reported date. A later verified update correcting an earlier one is a correction,
not a conflict. Both scenarios are pinned in the evaluation suite, and **false
escalations carry the same weight as missed ones** — a triage system that cries
wolf fails by a slower route.

### Restraint

**"Healthy cover · success"** — same 5-day delay, ample stock: **15.1** days of
cover, gap **0**, `risk_level: low`, action `monitor`, quantity **0**. Expediting
every late shipment is the expensive failure mode, and it is the one a naive
system commits.

### The evaluation

`python -m avathon.evaluation` runs eight cases, balanced between must-stop and
must-proceed, and writes `results/eval_summary.md`. Headline: 0 missed
escalations, 0 false escalations, 0 actions without approval, 8/8 proposals within
policy bounds, 1 guardrail block.

One test (`test_a_broken_workflow_is_actually_caught_by_the_harness`)
monkeypatches the policy engine to emit a quantity one unit over the cap and
asserts the suite fails. An evaluation that cannot fail is not evidence.

---

## A note on the figures above

They are the values at a 28-day lookback. The model chooses its own lookback
within an approved 7–56 day window, so with a live planner you may see, for
example, 21 days → 1.8 days of cover and a gap of 806. The **estimate** moves
because the model is exercising real latitude; the **bounds and the outcome** do
not, because they are computed by the policy engine. That difference is the
design.

## What to look at in the code

| Question | File |
| --- | --- |
| Does the model actually decide anything? | `src/avathon/llm.py` — the planner loop |
| What stops it? | `src/avathon/guardrails.py` — allow-list, schema, bounds, scope |
| Where do the numbers come from? | `src/avathon/policy.py` — no model calls in this file |
| How is it wired? | `src/avathon/workflow.py` — the interrupt is on line ~`interrupt_before` |
| Is it measured? | `src/avathon/evaluation.py`, `docs/evaluation.md` |
