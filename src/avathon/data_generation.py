"""Create reproducible, deliberately imperfect supply-chain operating data."""

from __future__ import annotations

import argparse
import csv
import random
from datetime import date, timedelta
from pathlib import Path


SKUS = [
    ("SKU-CRITICAL", 110, 35, 500),
    ("SKU-STABLE", 65, 18, 300),
    ("SKU-VOLATILE", 45, 25, 240),
]
SITES = ["Pune-DC", "Mumbai-DC"]
SUPPLIERS = [("SUP-A", 8), ("SUP-B", 11)]


def _poisson_like(mean: float, rng: random.Random) -> int:
    """Gamma-Poisson approximation: over-dispersed daily demand without numpy."""
    intensity = rng.gammavariate(5.0, mean / 5.0)
    return max(0, round(rng.gauss(intensity, intensity**0.5)))


def generate(output_dir: Path, seed: int = 42, days: int = 70) -> None:
    """Write four CSVs and a manifest. Day 70 includes an injected supplier delay."""
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = date(2026, 6, 14)
    demand_rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []
    po_rows: list[dict[str, object]] = []
    shipment_rows: list[dict[str, object]] = []

    for sku, mean, _, safety_stock in SKUS:
        for site in SITES:
            for day_index in range(days):
                day = start + timedelta(days=day_index)
                weekly_factor = 1.18 if day.weekday() in (0, 1) else 0.92
                spike = 1.65 if sku == "SKU-VOLATILE" and day_index in (58, 59) else 1
                demand_rows.append(
                    {
                        "date": day.isoformat(),
                        "sku": sku,
                        "site": site,
                        "units": _poisson_like(mean * weekly_factor * spike, rng),
                    }
                )

            on_hand = safety_stock + (400 if site == "Mumbai-DC" else 60)
            if sku == "SKU-CRITICAL" and site == "Pune-DC":
                on_hand = 175
            inventory_rows.append(
                {
                    "as_of_date": (start + timedelta(days=days - 1)).isoformat(),
                    "sku": sku,
                    "site": site,
                    "on_hand": on_hand,
                    "safety_stock": safety_stock,
                    "on_order": 0,
                }
            )

    for index, (supplier, standard_lead_days) in enumerate(SUPPLIERS, start=1):
        sku = "SKU-CRITICAL" if supplier == "SUP-A" else "SKU-STABLE"
        po_id = f"PO-{index:03d}"
        expected = start + timedelta(days=days - (8 if supplier == "SUP-A" else 3))
        actual = expected + timedelta(days=5 if supplier == "SUP-A" else 0)
        po_rows.append(
            {
                "po_id": po_id,
                "supplier": supplier,
                "sku": sku,
                "destination_site": "Pune-DC",
                "quantity": 500,
                "created_date": (expected - timedelta(days=standard_lead_days)).isoformat(),
                "expected_date": expected.isoformat(),
                "status": "in_transit",
            }
        )
        shipment_rows.append(
            {
                "po_id": po_id,
                "supplier": supplier,
                "reported_date": (start + timedelta(days=days - 1)).isoformat(),
                "eta_date": actual.isoformat(),
                "shipment_status": "delayed" if supplier == "SUP-A" else "on_time",
                "source_quality": "verified",
            }
        )

    _write_csv(output_dir / "daily_demand.csv", demand_rows)
    _write_csv(output_dir / "inventory.csv", inventory_rows)
    _write_csv(output_dir / "purchase_orders.csv", po_rows)
    _write_csv(output_dir / "shipments.csv", shipment_rows)
    (output_dir / "manifest.txt").write_text(
        "seed=42\n"
        "demand=gamma-poisson approximation with weekday seasonality\n"
        "injected_event=SUP-A delayed 5 days for SKU-CRITICAL to Pune-DC\n"
        "known_limitations=synthetic data does not represent contracts, capacities, or ERP errors\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate supply-chain demo data")
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--days", type=int, default=70)
    args = parser.parse_args()
    generate(args.output_dir, seed=args.seed, days=args.days)
