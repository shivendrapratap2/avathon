# Synthetic data specification

`src/avathon/data_generation.py` produces a seeded, 70-day data set for three
SKUs across two distribution centres. It is designed for workflow evaluation, not
to support any claim of real-world forecasting accuracy.

| Table | Grain | Key fields |
| --- | --- | --- |
| `daily_demand.csv` | SKU, site, day | demand units |
| `inventory.csv` | SKU, site, as-of date | on hand, safety stock, on order |
| `purchase_orders.csv` | purchase order | supplier, quantity, expected arrival |
| `shipments.csv` | shipment status update | reported date, ETA, status, source quality |

Demand uses an over-dispersed gamma-Poisson approximation with weekday
seasonality. `SKU-CRITICAL` at `Pune-DC` starts *below* its safety stock, so the
injected delay compounds an existing shortfall rather than creating one — the
common real case, and the one that exposes a days-of-cover calculation that
ignores safety stock.

## Scenarios

Every scenario is the same seeded base with one controlled perturbation, so any
change in workflow outcome is attributable to that perturbation alone.

| Scenario | Perturbation | Required outcome |
| --- | --- | --- |
| `verified_delay` | SUP-A five days late into Pune-DC | Bounded transfer proposal, held at the human gate |
| `healthy_stock` | Same delay, ample on-hand cover | `monitor`; no stock movement |
| `multi_po_clean` | A second legitimate verified PO into the same site | **Not** an escalation |
| `superseded_update` | An earlier update corrected by a later verified one | **Not** an escalation |
| `conflicting_evidence` | Two simultaneous divergent updates for one PO, one unverified | Fail closed, no proposal |
| `no_evidence` | All SKU-CRITICAL POs removed | Fail closed on absent evidence |

The two "not an escalation" scenarios carry as much weight as the fail-closed
ones. A triage system that stops on ordinary operating complexity — several open
POs, a corrected ETA — produces alert fatigue and is switched off, which is a
safety failure by a slower route.

The manifest written alongside each data set records the seed, the scenario, its
intent, and the exact injected event.

## Why synthetic

The required joined grain — daily demand, inventory position, open purchase
orders, and successive shipment-status updates carrying provenance, at a common
SKU/site key — does not exist in a public dataset. Provenance in particular
(`source_quality`) is the field the fail-closed behaviour depends on, and it is
essentially never published.

## Limitations

The data does not represent supplier contracts, capacity constraints, allocation
or reservation logic, transport lead times between DCs, or ERP data-entry errors.
It validates workflow behaviour and traceability, not causal performance on a
live network. Calibration against real historical demand, and a shadow-mode
replay of historical ERP outcomes, should both precede any deployment.
