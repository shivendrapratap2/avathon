"""The planner layer: a model that genuinely decides, inside a hard boundary.

Each agent hands the model an objective and an allow-listed toolset. The model
chooses which tools to call, in what order, and with what lookback window. It
receives the results and decides whether it has enough evidence. Nothing about
that loop is scripted.

What the model never does: produce a number, choose an action, or reach a tool
without passing ``guardrails.ToolPolicy``. After the deterministic policy engine
computes the figures, the model is asked once more for a planner-facing
explanation, and that explanation is checked for numeric grounding before it is
allowed into the trace.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .guardrails import ToolCallSafetyError, ToolPolicy


DEFAULT_MODEL = "gpt-4o-mini"

PLANNER_INSTRUCTIONS = (
    "You are an evidence-gathering agent in a safety-critical supply-chain triage "
    "workflow. Call the read-only tools you need to satisfy the objective, then stop "
    "and reply with a one-sentence statement of what the evidence shows.\n"
    "Rules you must follow:\n"
    "- Only investigate the SKU and site named in the objective.\n"
    "- Never propose, request, or describe an operational action; another component "
    "decides that, and a human approves it.\n"
    "- Never state a quantity, forecast, or risk level of your own; the figures are "
    "computed downstream from the tool results.\n"
    "- Do not call the same tool twice with the same arguments."
)

NARRATOR_INSTRUCTIONS = (
    "You explain a completed supply-chain analysis to an experienced planner.\n"
    "Rules you must follow:\n"
    "- Use ONLY the numbers present in the supplied facts. Never compute, round, "
    "estimate, or infer a new figure.\n"
    "- Two or three sentences. Plain operational language, no bullet points.\n"
    "- State what the evidence shows and what it means for stock cover. Do not add "
    "recommendations beyond the action already recorded in the facts.\n"
    "- If a fact is absent, omit it rather than guessing."
)

#: Rows echoed back to the model per tool call. Enough to reason over, bounded
#: so a wide query cannot inflate context or cost.
MAX_ROWS_TO_MODEL = 60


@dataclass(frozen=True)
class PlannerStep:
    """One auditable event inside a planner run."""

    kind: str  # tool_call | tool_result | guardrail_block | narrative
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    query_id: str = ""
    row_count: int = 0
    text: str = ""
    error: str = ""
    violation: str = ""
    call_id: str = ""
    response_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if value not in ("", 0, {})}


@dataclass
class PlannerRun:
    """Everything one agent's planner did, and everything it retrieved."""

    model: str
    steps: list[PlannerStep] = field(default_factory=list)
    evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    response_ids: list[str] = field(default_factory=list)
    closing_statement: str = ""

    @property
    def called_tools(self) -> set[str]:
        return set(self.evidence)

    @property
    def tool_call_count(self) -> int:
        return sum(1 for step in self.steps if step.kind == "tool_call")


class Planner(Protocol):
    """Anything that can drive an agent's evidence gathering."""

    model: str

    def gather(
        self,
        *,
        objective: str,
        policy: ToolPolicy,
        execute: Callable[[str, dict[str, Any]], dict[str, Any]],
    ) -> PlannerRun: ...

    def narrate(self, *, run: PlannerRun, facts: dict[str, Any]) -> tuple[str, list[str]]: ...


# -- numeric grounding -------------------------------------------------------

_NUMBER = re.compile(r"\d+(?:\.\d+)?")

#: Small integers used as ordinary prose ("two sites", "5 days") are permitted
#: only when they appear in the facts; these are counting words that carry no
#: operational claim and are allowed unconditionally.
_ALWAYS_ALLOWED = {"0", "1", "2", "3"}


#: Fields holding opaque identifiers rather than operational figures. Query IDs
#: are hex digests: harvesting digits out of them would whitelist arbitrary
#: numbers and quietly defeat the check.
OPAQUE_FIELDS = frozenset({
    "query_id", "evidence_chain", "evidence_refs", "response_ids", "call_id",
    "trace_id", "scenario_id", "narrative_grounding", "rollback", "reason",
    "constraints", "assumptions",
})

_HEX_DIGEST = re.compile(r"[0-9a-f]{8,}")


def _variants(raw: str) -> set[str]:
    """All spellings of one number: '08' and '8' and '8.0' are the same figure."""
    forms = {raw}
    try:
        value = float(raw)
    except ValueError:  # pragma: no cover - regex guarantees parseability
        return forms
    forms.add(str(value))
    if value.is_integer():
        forms.add(str(int(value)))
    return forms


def _allowed_numbers(node: Any) -> set[str]:
    """Collect every figure the policy engine actually computed."""
    allowed: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key in OPAQUE_FIELDS:
                continue
            allowed |= _allowed_numbers(value)
    elif isinstance(node, (list, tuple)):
        for item in node:
            allowed |= _allowed_numbers(item)
    elif isinstance(node, bool) or node is None:
        return allowed
    elif isinstance(node, (int, float)):
        allowed |= _variants(str(node))
    elif isinstance(node, str):
        if not _HEX_DIGEST.search(node):
            for raw in _NUMBER.findall(node):
                allowed |= _variants(raw)
    return allowed


def check_numeric_grounding(narrative: str, facts: dict[str, Any]) -> list[str]:
    """Return numbers the model used that the policy engine never computed.

    This is the guardrail on explanation. Constraining the decision path is not
    enough: a fluent, wrong number in a planner-facing summary is exactly the
    failure mode that erodes trust in an assistive system.

    Only computed figures count as grounding. Identifiers, prose fields the model
    was shown, and hex query IDs are excluded, because a check whose whitelist
    grows with every hash is not a check.
    """
    allowed = _allowed_numbers(facts) | _ALWAYS_ALLOWED
    ungrounded = [
        raw for raw in _NUMBER.findall(narrative) if not _variants(raw) & allowed
    ]
    return sorted(set(ungrounded), key=ungrounded.index)


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    rows = result.get("rows", [])
    return {
        "query_id": result.get("query_id"),
        "row_count": len(rows),
        "rows": rows[:MAX_ROWS_TO_MODEL],
        "truncated": len(rows) > MAX_ROWS_TO_MODEL,
    }


# -- live model planner ------------------------------------------------------


class LLMPlanner:
    """Drives evidence gathering through the OpenAI Responses API.

    Conversation state is held locally rather than through ``previous_response_id``
    so that ``store=False`` holds end to end: no operational evidence is retained
    on the provider side, and the full exchange stays reproducible from the trace.
    """

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, *, client: Any = None):
        if client is not None:
            self.client = client
        else:
            try:
                from openai import OpenAI
            except ImportError as error:  # pragma: no cover - environment guard
                raise RuntimeError(
                    "OpenAI SDK is not installed. Run `pip install -r requirements.txt`."
                ) from error
            self.client = OpenAI(api_key=api_key)
        self.model = model

    def gather(
        self,
        *,
        objective: str,
        policy: ToolPolicy,
        execute: Callable[[str, dict[str, Any]], dict[str, Any]],
    ) -> PlannerRun:
        run = PlannerRun(model=self.model)
        conversation: list[Any] = [{"role": "user", "content": objective}]

        for _ in range(policy.max_steps):
            response = self.client.responses.create(
                model=self.model,
                instructions=PLANNER_INSTRUCTIONS,
                input=conversation,
                tools=policy.tool_specs,
                tool_choice="auto",
                store=False,
            )
            run.response_ids.append(response.id)
            calls = [item for item in response.output if getattr(item, "type", "") == "function_call"]

            if not calls:
                run.closing_statement = (getattr(response, "output_text", "") or "").strip()
                run.steps.append(
                    PlannerStep(kind="narrative", text=run.closing_statement, response_id=response.id)
                )
                break

            for call in calls:
                conversation.append(_as_input_item(call))
                try:
                    arguments = json.loads(call.arguments or "{}")
                except json.JSONDecodeError as error:
                    raise ToolCallSafetyError(
                        "Model supplied malformed tool arguments.",
                        violation="malformed_arguments",
                        tool_name=call.name,
                    ) from error

                run.steps.append(
                    PlannerStep(
                        kind="tool_call",
                        tool_name=call.name,
                        arguments=arguments,
                        call_id=call.call_id,
                        response_id=response.id,
                    )
                )
                clean = policy.validate(call.name, arguments)
                result = execute(call.name, clean)
                run.evidence[call.name] = result
                run.steps.append(
                    PlannerStep(
                        kind="tool_result",
                        tool_name=call.name,
                        arguments=clean,
                        query_id=result["query_id"],
                        row_count=len(result["rows"]),
                        call_id=call.call_id,
                    )
                )
                conversation.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(_compact(result), default=str),
                    }
                )

        policy.check_required_evidence(run.called_tools)
        return run

    def narrate(self, *, run: PlannerRun, facts: dict[str, Any]) -> tuple[str, list[str]]:
        response = self.client.responses.create(
            model=self.model,
            instructions=NARRATOR_INSTRUCTIONS,
            input=[
                {
                    "role": "user",
                    "content": (
                        "Explain this completed analysis to a planner using only these "
                        "facts:\n" + json.dumps(facts, default=str, indent=2)
                    ),
                }
            ],
            store=False,
        )
        run.response_ids.append(response.id)
        narrative = (getattr(response, "output_text", "") or "").strip()
        return narrative, check_numeric_grounding(narrative, facts)


def _as_input_item(call: Any) -> dict[str, Any]:
    """Echo a model function call back into the conversation without the SDK object."""
    if hasattr(call, "model_dump"):
        item = call.model_dump(exclude_none=True)
        item.pop("status", None)
        return item
    return {  # pragma: no cover - defensive, for hand-built fakes
        "type": "function_call",
        "name": call.name,
        "arguments": call.arguments,
        "call_id": call.call_id,
    }


# -- deterministic stand-in --------------------------------------------------


class ScriptedPlanner:
    """A planner that follows a fixed script instead of calling a model.

    Two uses, both deliberate. Tests exercise the graph and the guardrails without
    network access or spend. The console's guardrail probe uses a deliberately
    misbehaving script to demonstrate that a rogue tool call is blocked before it
    reaches the database — a live model cannot be relied on to misbehave on cue.
    """

    def __init__(
        self,
        script: list[tuple[str, dict[str, Any]]],
        *,
        model: str = "scripted-planner",
        closing_statement: str = "Evidence gathered.",
        narrative: str = "",
    ):
        self.script = script
        self.model = model
        self.closing_statement = closing_statement
        self._narrative = narrative

    def gather(
        self,
        *,
        objective: str,
        policy: ToolPolicy,
        execute: Callable[[str, dict[str, Any]], dict[str, Any]],
    ) -> PlannerRun:
        run = PlannerRun(model=self.model)
        for index, (tool_name, arguments) in enumerate(self.script[: policy.max_steps]):
            run.steps.append(
                PlannerStep(
                    kind="tool_call", tool_name=tool_name, arguments=arguments,
                    call_id=f"scripted-{index}",
                )
            )
            clean = policy.validate(tool_name, arguments)
            result = execute(tool_name, clean)
            run.evidence[tool_name] = result
            run.steps.append(
                PlannerStep(
                    kind="tool_result", tool_name=tool_name, arguments=clean,
                    query_id=result["query_id"], row_count=len(result["rows"]),
                    call_id=f"scripted-{index}",
                )
            )
        run.closing_statement = self.closing_statement
        policy.check_required_evidence(run.called_tools)
        return run

    def narrate(self, *, run: PlannerRun, facts: dict[str, Any]) -> tuple[str, list[str]]:
        if not self._narrative:
            return "", []
        return self._narrative, check_numeric_grounding(self._narrative, facts)


def reference_planner_factory(sku: str, site: str, lookback_days: int = 28):
    """Scripted planners that take the shortest compliant path through each agent.

    This is the control arm. Running the same scenarios against it and against a
    live model isolates workflow behaviour from model behaviour, and lets the
    evaluation suite run in CI with no API key and no spend.
    """
    scripts: dict[str, list[tuple[str, dict[str, Any]]]] = {
        "risk_agent": [("supplier_exposure", {"sku": sku, "site": site})],
        "impact_agent": [
            ("demand_history", {"sku": sku, "site": site, "lookback_days": lookback_days})
        ],
        "replenishment_agent": [
            ("alternate_site_availability", {"sku": sku, "exclude_site": site})
        ],
    }

    def factory(agent: str) -> ScriptedPlanner:
        return ScriptedPlanner(
            scripts[agent],
            model="reference-plan",
            closing_statement=f"{agent} gathered its required evidence.",
        )

    return factory


def rogue_planner_factory(sku: str, site: str, *, rogue_agent: str = "risk_agent"):
    """A planner that attempts an out-of-scope tool call at one agent.

    Used by the console's guardrail probe. A live model cannot be relied on to
    misbehave on demand, so demonstrating that the boundary holds requires a
    deliberate offender.
    """
    reference = reference_planner_factory(sku, site)

    def factory(agent: str):
        if agent != rogue_agent:
            return reference(agent)
        return ScriptedPlanner(
            [("supplier_exposure", {"sku": "SKU-VOLATILE", "site": "Mumbai-DC"})],
            model="rogue-plan",
            closing_statement="Attempted an out-of-scope query.",
        )

    return factory
