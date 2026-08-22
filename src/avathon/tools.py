"""Bounded, read-only operational analytics tools available to agents."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import duckdb


class SupplyChainAnalyticsTool:
    """A parameterized, query-logged DuckDB tool with no write capability.

    Agents can select an allow-listed operation, but never provide arbitrary SQL.
    This prevents a hallucinated tool call from becoming an unbounded database
    operation or a data mutation.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.query_log: list[dict[str, Any]] = []

    def supplier_exposure(self, sku: str, site: str) -> dict[str, Any]:
        """Return delayed inbound orders and their destination inventory snapshot."""
        connection = duckdb.connect(database=":memory:", read_only=False)
        try:
            for table in ("purchase_orders", "shipments", "inventory"):
                path = (self.data_dir / f"{table}.csv").as_posix().replace("'", "''")
                connection.execute(
                    f"CREATE VIEW {table} AS SELECT * FROM read_csv_auto('{path}')"
                )
            cursor = connection.execute(
                """
                SELECT
                    po.po_id, po.supplier, po.sku, po.destination_site, po.quantity,
                    po.expected_date, shipment.eta_date, shipment.shipment_status,
                    shipment.source_quality, inventory.on_hand, inventory.safety_stock,
                    date_diff('day', CAST(po.expected_date AS DATE), CAST(shipment.eta_date AS DATE))
                        AS delay_days
                FROM purchase_orders AS po
                JOIN shipments AS shipment ON po.po_id = shipment.po_id
                JOIN inventory
                    ON po.sku = inventory.sku AND po.destination_site = inventory.site
                WHERE po.sku = ? AND po.destination_site = ?
                ORDER BY po.po_id, shipment.reported_date
                """,
                [sku, site],
            )
            columns = [item[0] for item in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        finally:
            connection.close()
        query_id = "duckdb:" + hashlib.sha256(
            f"supplier_exposure:{sku}:{site}:{len(self.query_log)}".encode()
        ).hexdigest()[:12]
        self.query_log.append(
            {"query_id": query_id, "operation": "supplier_exposure", "sku": sku, "site": site}
        )
        return {"query_id": query_id, "rows": rows}

    def demand_history(self, sku: str, site: str, lookback_days: int = 28) -> dict[str, Any]:
        """Return recent daily demand. Lookback is clamped to an approved range."""
        if not 7 <= lookback_days <= 56:
            raise ValueError("lookback_days must be between 7 and 56")
        connection = duckdb.connect(database=":memory:", read_only=False)
        try:
            path = (self.data_dir / "daily_demand.csv").as_posix().replace("'", "''")
            connection.execute(f"CREATE VIEW demand AS SELECT * FROM read_csv_auto('{path}')")
            cursor = connection.execute(
                """
                SELECT date, units
                FROM demand
                WHERE sku = ? AND site = ?
                ORDER BY CAST(date AS DATE) DESC
                LIMIT ?
                """,
                [sku, site, lookback_days],
            )
            columns = [item[0] for item in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        finally:
            connection.close()
        query_id = "duckdb:" + hashlib.sha256(
            f"demand_history:{sku}:{site}:{lookback_days}:{len(self.query_log)}".encode()
        ).hexdigest()[:12]
        self.query_log.append(
            {"query_id": query_id, "operation": "demand_history", "sku": sku, "site": site}
        )
        return {"query_id": query_id, "rows": rows}
