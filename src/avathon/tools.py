"""Bounded, read-only operational analytics tools available to agents.

Agents select an allow-listed operation by name; they never supply SQL. Every
call is parameterized, logged, and returned with an immutable ``query_id`` that
downstream messages cite as evidence.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import duckdb


MIN_LOOKBACK_DAYS = 7
MAX_LOOKBACK_DAYS = 56


# JSON-schema specifications published to the model. The runtime allow-list in
# guardrails.py is the enforcement point; this list is only what the model sees.
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "supplier_exposure",
        "description": (
            "Read-only. Retrieve open purchase orders, their shipment status "
            "updates, and the destination inventory snapshot for one SKU at one "
            "site. Use this to establish whether a delay exists and how trustworthy "
            "the shipment evidence is."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sku": {"type": "string", "description": "Stock keeping unit under investigation."},
                "site": {"type": "string", "description": "Destination distribution centre."},
            },
            "required": ["sku", "site"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "demand_history",
        "description": (
            "Read-only. Retrieve recent daily demand units for one SKU at one site. "
            "Choose a lookback window between 7 and 56 days that suits the volatility "
            "of the SKU."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sku": {"type": "string"},
                "site": {"type": "string"},
                "lookback_days": {
                    "type": "integer",
                    "description": f"Days of history, {MIN_LOOKBACK_DAYS}-{MAX_LOOKBACK_DAYS}.",
                },
            },
            "required": ["sku", "site", "lookback_days"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "alternate_site_availability",
        "description": (
            "Read-only. Retrieve on-hand and safety-stock levels for one SKU at every "
            "site EXCEPT the one named in exclude_site. Use this before proposing an "
            "inter-site transfer, so the proposed quantity is grounded in real surplus."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sku": {"type": "string"},
                "exclude_site": {
                    "type": "string",
                    "description": "The site with the shortage; it is omitted from results.",
                },
            },
            "required": ["sku", "exclude_site"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


class SupplyChainAnalyticsTool:
    """A parameterized, query-logged DuckDB tool with no write capability.

    A hallucinated tool call can therefore never become an unbounded database
    operation or a data mutation. It can only become a rejected call.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.query_log: list[dict[str, Any]] = []
        self.result_store: dict[str, list[dict[str, Any]]] = {}

    # -- internals ---------------------------------------------------------

    def _connect(self, tables: tuple[str, ...]) -> duckdb.DuckDBPyConnection:
        connection = duckdb.connect(database=":memory:", read_only=False)
        for table in tables:
            path = (self.data_dir / f"{table}.csv").as_posix().replace("'", "''")
            connection.execute(f"CREATE VIEW {table} AS SELECT * FROM read_csv_auto('{path}')")
        return connection

    def _record(self, operation: str, arguments: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
        signature = ":".join(f"{key}={arguments[key]}" for key in sorted(arguments))
        query_id = "duckdb:" + hashlib.sha256(
            f"{operation}:{signature}:{len(self.query_log)}".encode()
        ).hexdigest()[:12]
        self.query_log.append({"query_id": query_id, "operation": operation, "arguments": arguments})
        self.result_store[query_id] = rows
        return {"query_id": query_id, "operation": operation, "rows": rows}

    @staticmethod
    def _rows(cursor: Any) -> list[dict[str, Any]]:
        columns = [item[0] for item in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    # -- allow-listed operations ------------------------------------------

    def supplier_exposure(self, sku: str, site: str) -> dict[str, Any]:
        """Return every shipment update for open POs into one site, plus inventory.

        All updates are returned, not a deduplicated view: evidence-integrity
        checks in policy.py need to see contradictory updates for the same PO.
        """
        connection = self._connect(("purchase_orders", "shipments", "inventory"))
        try:
            cursor = connection.execute(
                """
                SELECT
                    po.po_id, po.supplier, po.sku, po.destination_site, po.quantity,
                    po.expected_date, shipment.reported_date, shipment.eta_date,
                    shipment.shipment_status, shipment.source_quality,
                    inventory.on_hand, inventory.safety_stock, inventory.on_order,
                    date_diff('day', CAST(po.expected_date AS DATE), CAST(shipment.eta_date AS DATE))
                        AS delay_days
                FROM purchase_orders AS po
                JOIN shipments AS shipment ON po.po_id = shipment.po_id
                JOIN inventory
                    ON po.sku = inventory.sku AND po.destination_site = inventory.site
                WHERE po.sku = ? AND po.destination_site = ?
                ORDER BY po.po_id, shipment.reported_date, shipment.eta_date
                """,
                [sku, site],
            )
            rows = self._rows(cursor)
        finally:
            connection.close()
        return self._record("supplier_exposure", {"sku": sku, "site": site}, rows)

    def demand_history(self, sku: str, site: str, lookback_days: int = 28) -> dict[str, Any]:
        """Return recent daily demand. Lookback is clamped to an approved range."""
        if not MIN_LOOKBACK_DAYS <= lookback_days <= MAX_LOOKBACK_DAYS:
            raise ValueError(
                f"lookback_days must be between {MIN_LOOKBACK_DAYS} and {MAX_LOOKBACK_DAYS}"
            )
        connection = self._connect(("daily_demand",))
        try:
            cursor = connection.execute(
                """
                SELECT date, units
                FROM daily_demand
                WHERE sku = ? AND site = ?
                ORDER BY CAST(date AS DATE) DESC
                LIMIT ?
                """,
                [sku, site, lookback_days],
            )
            rows = self._rows(cursor)
        finally:
            connection.close()
        return self._record(
            "demand_history", {"sku": sku, "site": site, "lookback_days": lookback_days}, rows
        )

    def alternate_site_availability(self, sku: str, exclude_site: str) -> dict[str, Any]:
        """Return inventory at every other site holding this SKU."""
        connection = self._connect(("inventory",))
        try:
            cursor = connection.execute(
                """
                SELECT site, on_hand, safety_stock, on_order,
                       GREATEST(on_hand - safety_stock, 0) AS transferable_surplus
                FROM inventory
                WHERE sku = ? AND site <> ?
                ORDER BY transferable_surplus DESC
                """,
                [sku, exclude_site],
            )
            rows = self._rows(cursor)
        finally:
            connection.close()
        return self._record(
            "alternate_site_availability", {"sku": sku, "exclude_site": exclude_site}, rows
        )

    # -- dispatch ----------------------------------------------------------

    @property
    def operations(self) -> dict[str, Any]:
        return {
            "supplier_exposure": self.supplier_exposure,
            "demand_history": self.demand_history,
            "alternate_site_availability": self.alternate_site_availability,
        }

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run an allow-listed operation. Unknown names raise rather than guess."""
        if tool_name not in self.operations:
            raise KeyError(f"'{tool_name}' is not an allow-listed operation.")
        return self.operations[tool_name](**arguments)
