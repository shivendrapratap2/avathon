"""Create reproducible, deliberately imperfect supply-chain operating data.

Each scenario is a controlled perturbation of the same seeded base, so a change
in workflow outcome is attributable to the perturbation and nothing else.
"""

from __future__ import annotations

import argparse
import csv
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any


# (sku, mean daily demand, safety stock)
SKUS = [
    ("SKU-CRITICAL", 110, 500),
    ("SKU-STABLE", 65, 300),
    ("SKU-VOLATILE", 45, 240),
]
SITES = ["Pune-DC", "Mumbai-DC"]
SUPPLIERS = [("SUP-A", 8), ("SUP-B", 11)]

SCENARIOS: dict[str, str] = {
    "verified_delay": (
        "A single verified five-day delay on a SKU already below safety stock. "
        "Expected: a bounded transfer proposal held at the human gate."
    ),
    "conflicting_evidence": (
        "Two simultaneous, contradictory updates for the same PO, one unverified. "
        "Expected: fail closed, no proposal."
    ),
    "multi_po_clean": (
        "Two separate, legitimately different, verified POs into the same site. "
        "Expected: NOT an escalation - this is the normal operating case."
    ),
    "superseded_update": (
        "An earlier update corrected by a later verified one for the same PO. "
        "Expected: NOT an escalation - revision is not contradiction."
    ),
    "no_evidence": (
        "No purchase order exists for the alerted SKU and site. "
        "Expected: fail closed on absent evidence."
    ),
    "healthy_stock": (
        "A verified delay against ample on-hand cover. "
        "Expected: a monitor recommendation, no stock movement."
    ),
}


def _poisson_like(mean: float, rng: random.Random) -> int:
    """Gamma-Poisson approximation: over-dispersed daily demand without numpy."""
    intensity = rng.gammavariate(5.0, mean / 5.0)
    return max(0, round(rng.gauss(intensity, intensity**0.5)))


def generate(
    output_dir: Path, seed: int = 42, days: int = 70, scenario: str = "verified_delay"
) -> None:
    """Write four CSVs and a manifest describing exactly what was injected."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from {sorted(SCENARIOS)}.")

    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = date(2026, 6, 14)
    last_day = start + timedelta(days=days - 1)

    demand_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    po_rows: list[dict[str, Any]] = []
    shipment_rows: list[dict[str, Any]] = []

    for sku, mean, safety_stock in SKUS:
        for site in SITES:
            for day_index in range(days):
                day = start + timedelta(days=day_index)
                weekly_factor = 1.18 if day.weekday() in (0, 1) else 0.92
                spike = 1.65 if sku == "SKU-VOLATILE" and day_index in (58, 59) else 1
                demand_rows.append({
                    "date": day.isoformat(), "sku": sku, "site": site,
                    "units": _poisson_like(mean * weekly_factor * spike, rng),
                })

            on_hand = safety_stock + (400 if site == "Mumbai-DC" else 60)
            if sku == "SKU-CRITICAL" and site == "Pune-DC":
                # Deliberately below safety stock: the delay compounds an existing shortfall.
                on_hand = 175
                if scenario == "healthy_stock":
                    on_hand = 1400
            inventory_rows.append({
                "as_of_date": last_day.isoformat(), "sku": sku, "site": site,
                "on_hand": on_hand, "safety_stock": safety_stock, "on_order": 0,
            })

    for index, (supplier, standard_lead_days) in enumerate(SUPPLIERS, start=1):
        sku = "SKU-CRITICAL" if supplier == "SUP-A" else "SKU-STABLE"
        po_id = f"PO-{index:03d}"
        expected = last_day - timedelta(days=1 if supplier == "SUP-A" else 0)
        actual = expected + timedelta(days=5 if supplier == "SUP-A" else 0)
        po_rows.append({
            "po_id": po_id, "supplier": supplier, "sku": sku,
            "destination_site": "Pune-DC", "quantity": 500,
            "created_date": (expected - timedelta(days=standard_lead_days)).isoformat(),
            "expected_date": expected.isoformat(), "status": "in_transit",
        })
        shipment_rows.append({
            "po_id": po_id, "supplier": supplier,
            "reported_date": last_day.isoformat(), "eta_date": actual.isoformat(),
            "shipment_status": "delayed" if supplier == "SUP-A" else "on_time",
            "source_quality": "verified",
        })

    injected = _inject(scenario, po_rows, shipment_rows, start, days, last_day)

    _write_csv(output_dir / "daily_demand.csv", demand_rows)
    _write_csv(output_dir / "inventory.csv", inventory_rows)
    _write_csv(output_dir / "purchase_orders.csv", po_rows)
    _write_csv(output_dir / "shipments.csv", shipment_rows)
    (output_dir / "manifest.txt").write_text(
        f"seed={seed}\n"
        f"days={days}\n"
        "demand=gamma-poisson approximation with weekday seasonality\n"
        f"scenario={scenario}\n"
        f"scenario_intent={SCENARIOS[scenario]}\n"
        f"injected_event={injected}\n"
        "known_limitations=synthetic data does not represent contracts, capacities, "
        "supplier behaviour, or ERP data-entry errors\n",
        encoding="utf-8",
    )


def _inject(
    scenario: str,
    po_rows: list[dict[str, Any]],
    shipment_rows: list[dict[str, Any]],
    start: date,
    days: int,
    last_day: date,
) -> str:
    """Apply the scenario-specific perturbation. Returns a manifest description."""
    if scenario in ("verified_delay", "healthy_stock"):
        return "SUP-A delayed 5 days for SKU-CRITICAL into Pune-DC"

    if scenario == "conflicting_evidence":
        # Same PO, same reported date, contradictory ETA and status, unverified source.
        shipment_rows.append({
            "po_id": "PO-001", "supplier": "SUP-A",
            "reported_date": last_day.isoformat(),
            "eta_date": (last_day + timedelta(days=1)).isoformat(),
            "shipment_status": "on_time", "source_quality": "unverified",
        })
        return "second simultaneous unverified update for PO-001 contradicting the first"

    if scenario == "superseded_update":
        # An older, stale update that a later verified one already corrected.
        shipment_rows.insert(0, {
            "po_id": "PO-001", "supplier": "SUP-A",
            "reported_date": (last_day - timedelta(days=3)).isoformat(),
            "eta_date": (last_day - timedelta(days=1)).isoformat(),
            "shipment_status": "on_time", "source_quality": "verified",
        })
        return "earlier PO-001 update superseded by a later verified update"

    if scenario == "multi_po_clean":
        expected = last_day + timedelta(days=9)
        po_rows.append({
            "po_id": "PO-003", "supplier": "SUP-B", "sku": "SKU-CRITICAL",
            "destination_site": "Pune-DC", "quantity": 400,
            "created_date": (expected - timedelta(days=11)).isoformat(),
            "expected_date": expected.isoformat(), "status": "in_transit",
        })
        shipment_rows.append({
            "po_id": "PO-003", "supplier": "SUP-B",
            "reported_date": last_day.isoformat(), "eta_date": expected.isoformat(),
            "shipment_status": "on_time", "source_quality": "verified",
        })
        return "second legitimate verified PO-003 into Pune-DC, on time"

    if scenario == "no_evidence":
        po_rows[:] = [row for row in po_rows if row["sku"] != "SKU-CRITICAL"]
        shipment_rows[:] = [row for row in shipment_rows if row["po_id"] != "PO-001"]
        return "all SKU-CRITICAL purchase orders removed"

    return "none"  # pragma: no cover - guarded by the scenario check above


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate supply-chain demo data")
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--days", type=int, default=70)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="verified_delay")
    args = parser.parse_args()
    generate(args.output_dir, seed=args.seed, days=args.days, scenario=args.scenario)
