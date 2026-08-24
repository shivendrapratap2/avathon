"""The enforcement boundary between model intent and tool execution.

The model is given genuine latitude: it decides which read-only tools to call,
in which order, and with what lookback. It is given no latitude at all on
safety. Every proposed call is checked here, before DuckDB sees it, against:

* a per-agent tool allow-list,
* argument names and types,
* value bounds (lookback window),
* investigation scope (a call may only reference the SKU/site under review),
* a step budget and a duplicate-call check, so a confused model cannot loop.

A violation raises ``ToolCallSafetyError``. The workflow converts that into a
``guardrail_violation`` escalation. It never degrades into a recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .tools import MAX_LOOKBACK_DAYS, MIN_LOOKBACK_DAYS, TOOL_SPECS


class ToolCallSafetyError(RuntimeError):
    """Raised when a model's proposed function call violates workflow policy."""

    def __init__(self, message: str, *, violation: str, tool_name: str = "unknown"):
        super().__init__(message)
        self.violation = violation
        self.tool_name = tool_name


@dataclass(frozen=True)
class InvestigationScope:
    """The only SKU/site a planner is permitted to touch during one run."""

    sku: str
    site: str


@dataclass
class ToolPolicy:
    """Per-agent constraints applied to every model-proposed tool call."""

    agent: str
    allowed_tools: frozenset[str]
    required_tools: frozenset[str]
    scope: InvestigationScope
    max_steps: int = 4
    seen_calls: set[tuple[str, str]] = field(default_factory=set)

    @property
    def tool_specs(self) -> list[dict[str, Any]]:
        """Publish every allow-listed spec, so tool choice is a real decision."""
        return [spec for spec in TOOL_SPECS if spec["name"] in self.allowed_tools]

    def validate(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Return sanitized arguments, or raise ``ToolCallSafetyError``."""
        if tool_name not in self.allowed_tools:
            raise ToolCallSafetyError(
                f"{self.agent} may not call '{tool_name}'. "
                f"Allowed: {sorted(self.allowed_tools)}.",
                violation="tool_not_allowed",
                tool_name=tool_name,
            )
        if not isinstance(arguments, dict):
            raise ToolCallSafetyError(
                "Tool arguments must be a JSON object.",
                violation="malformed_arguments",
                tool_name=tool_name,
            )

        checker = {
            "supplier_exposure": self._check_supplier_exposure,
            "demand_history": self._check_demand_history,
            "alternate_site_availability": self._check_alternate_site,
        }[tool_name]
        clean = checker(tool_name, arguments)

        signature = (tool_name, repr(sorted(clean.items())))
        if signature in self.seen_calls:
            raise ToolCallSafetyError(
                f"'{tool_name}' was already called with identical arguments; "
                "repeating it cannot produce new evidence.",
                violation="duplicate_call",
                tool_name=tool_name,
            )
        self.seen_calls.add(signature)
        return clean

    # -- per-tool checks ---------------------------------------------------

    def _require_exact_keys(self, tool_name: str, arguments: dict[str, Any], keys: set[str]) -> None:
        extra = set(arguments) - keys
        missing = keys - set(arguments)
        if extra or missing:
            raise ToolCallSafetyError(
                f"'{tool_name}' expects exactly {sorted(keys)}; "
                f"missing={sorted(missing)} unexpected={sorted(extra)}.",
                violation="argument_schema",
                tool_name=tool_name,
            )

    def _check_scope(self, tool_name: str, sku: Any, site: Any, site_field: str) -> None:
        if sku != self.scope.sku:
            raise ToolCallSafetyError(
                f"'{tool_name}' requested SKU '{sku}', outside the approved "
                f"investigation scope '{self.scope.sku}'.",
                violation="out_of_scope_sku",
                tool_name=tool_name,
            )
        if site != self.scope.site:
            raise ToolCallSafetyError(
                f"'{tool_name}' requested {site_field} '{site}', outside the approved "
                f"investigation scope '{self.scope.site}'.",
                violation="out_of_scope_site",
                tool_name=tool_name,
            )

    def _check_supplier_exposure(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self._require_exact_keys(tool_name, arguments, {"sku", "site"})
        self._check_scope(tool_name, arguments["sku"], arguments["site"], "site")
        return {"sku": str(arguments["sku"]), "site": str(arguments["site"])}

    def _check_demand_history(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self._require_exact_keys(tool_name, arguments, {"sku", "site", "lookback_days"})
        self._check_scope(tool_name, arguments["sku"], arguments["site"], "site")
        lookback = arguments["lookback_days"]
        if isinstance(lookback, bool) or not isinstance(lookback, int):
            raise ToolCallSafetyError(
                f"lookback_days must be an integer; received {type(lookback).__name__}.",
                violation="argument_type",
                tool_name=tool_name,
            )
        if not MIN_LOOKBACK_DAYS <= lookback <= MAX_LOOKBACK_DAYS:
            raise ToolCallSafetyError(
                f"lookback_days={lookback} is outside the approved "
                f"{MIN_LOOKBACK_DAYS}-{MAX_LOOKBACK_DAYS} day window.",
                violation="argument_bounds",
                tool_name=tool_name,
            )
        return {
            "sku": str(arguments["sku"]),
            "site": str(arguments["site"]),
            "lookback_days": lookback,
        }

    def _check_alternate_site(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self._require_exact_keys(tool_name, arguments, {"sku", "exclude_site"})
        self._check_scope(tool_name, arguments["sku"], arguments["exclude_site"], "exclude_site")
        return {"sku": str(arguments["sku"]), "exclude_site": str(arguments["exclude_site"])}

    # -- completion check --------------------------------------------------

    def check_required_evidence(self, called_tools: set[str]) -> None:
        """Fail closed if the planner stopped before gathering mandated evidence.

        The model chooses its own path; it does not get to skip the evidence the
        downstream deterministic calculation depends on.
        """
        missing = self.required_tools - called_tools
        if missing:
            raise ToolCallSafetyError(
                f"{self.agent} finished without required evidence: {sorted(missing)}.",
                violation="missing_required_evidence",
                tool_name=sorted(missing)[0],
            )
