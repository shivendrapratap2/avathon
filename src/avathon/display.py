"""Row builders for the console's tables.

These live outside ``app.py`` so they can be tested without a browser. Streamlit
serializes tables through Arrow, which infers one type per column: a column
mixing ``int`` and ``str`` fails conversion and prints a pyarrow traceback to the
console. Every cell is therefore rendered to a string here, once, in a place a
test can reach.
"""

from __future__ import annotations

from typing import Any


PLACEHOLDER = "—"


def proposal_rows(proposal: dict[str, Any]) -> list[dict[str, str]]:
    """Field/value rows for the proposal shown at the human gate."""
    summary = {
        "Action": proposal["action"],
        "Quantity": f"{proposal['quantity']} units",
        "From": proposal.get("source_site") or PLACEHOLDER,
        "Risk level": proposal["risk_level"],
        "Safety-stock gap": f"{proposal['safety_stock_gap_units']} units",
        "Alternate surplus": f"{proposal['alternate_site_surplus_units']} units",
        "Policy cap": f"{proposal['policy_cap_units']} units",
    }
    return [{"Field": str(key), "Value": str(value)} for key, value in summary.items()]


def trace_rows(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """One row per typed message, for the trace overview table."""
    return [
        {
            "#": str(index),
            "From": str(message["sender"]),
            "To": str(message["recipient"]),
            "Type": str(message["message_type"]),
            "Confidence": f"{message['confidence']:.0%}",
            "Evidence": ", ".join(message["evidence_refs"]) or PLACEHOLDER,
        }
        for index, message in enumerate(messages, start=1)
    ]
