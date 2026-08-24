"""Constrained OpenAI Responses API tool calling for operational agents.

The model can request only one of the allow-listed read-only analytics tools.
All arguments are checked against the workflow context before a local tool is
executed. The module deliberately captures a short outcome summary rather than
hidden reasoning content.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable


class ToolCallSafetyError(RuntimeError):
    """Raised when a model's proposed function call violates workflow policy."""


@dataclass(frozen=True)
class ToolCallTrace:
    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    call_id: str
    request_id: str
    follow_up_id: str
    model: str
    summary: str


TOOLS = [
    {
        "type": "function",
        "name": "supplier_exposure",
        "description": "Retrieve read-only purchase-order, shipment, and destination inventory evidence for one SKU and site.",
        "parameters": {
            "type": "object",
            "properties": {
                "sku": {"type": "string"},
                "site": {"type": "string"},
            },
            "required": ["sku", "site"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "demand_history",
        "description": "Retrieve read-only recent daily demand for one SKU and site. Use only for demand-impact analysis.",
        "parameters": {
            "type": "object",
            "properties": {
                "sku": {"type": "string"},
                "site": {"type": "string"},
                "lookback_days": {"type": "integer", "minimum": 7, "maximum": 56},
            },
            "required": ["sku", "site", "lookback_days"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


class OpenAIToolCaller:
    """Executes a model-selected tool only after strict local validation."""

    def __init__(self, api_key: str, model: str):
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "OpenAI SDK is not installed. Run `python -m pip install -r requirements.txt`."
            ) from error
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def call(
        self,
        *,
        role: str,
        expected_tool: str,
        sku: str,
        site: str,
        execute: Callable[..., dict[str, Any]],
    ) -> ToolCallTrace:
        instruction = (
            f"You are the {role} in a safety-critical supply-chain workflow. "
            f"Choose the single correct read-only function for this task. The task is: {expected_tool}. "
            f"Use SKU '{sku}' and site '{site}'. Do not recommend or execute any operational action. "
            "A function call is required; do not answer from memory."
        )
        response = self.client.responses.create(
            model=self.model,
            instructions=instruction,
            input="Select the correct analytics tool and retrieve grounded evidence.",
            tools=TOOLS,
            tool_choice="required",
            store=False,
        )
        calls = [item for item in response.output if item.type == "function_call"]
        if len(calls) != 1:
            raise ToolCallSafetyError(f"Expected exactly one function call; received {len(calls)}.")
        call = calls[0]
        if call.name != expected_tool:
            raise ToolCallSafetyError(
                f"Model selected '{call.name}', but '{expected_tool}' is required for the {role}."
            )
        try:
            arguments = json.loads(call.arguments)
        except json.JSONDecodeError as error:
            raise ToolCallSafetyError("Model supplied malformed tool arguments.") from error
        if arguments.get("sku") != sku or arguments.get("site") != site:
            raise ToolCallSafetyError("Model tool arguments do not match the approved SKU/site context.")
        if expected_tool == "demand_history":
            if arguments.get("lookback_days") != 28:
                raise ToolCallSafetyError("Demand history must use the approved 28-day lookback.")
        elif set(arguments) != {"sku", "site"}:
            raise ToolCallSafetyError("Supplier exposure accepts only SKU and site.")

        result = execute(**arguments)
        follow_up = self.client.responses.create(
            model=self.model,
            previous_response_id=response.id,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result, default=str),
                }
            ],
            store=False,
        )
        return ToolCallTrace(
            tool_name=call.name,
            arguments=arguments,
            result=result,
            call_id=call.call_id,
            request_id=response.id,
            follow_up_id=follow_up.id,
            model=self.model,
            summary=(follow_up.output_text or "No summary returned.").strip(),
        )
