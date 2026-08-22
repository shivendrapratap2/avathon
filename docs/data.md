# Synthetic data specification

`src/avathon/data_generation.py` produces a seeded, 70-day data set for three
SKUs and two distribution centres. It is intentionally designed for workflow
evaluation, not to claim real-world forecasting accuracy.

| Table | Grain | Key fields |
| --- | --- | --- |
| `daily_demand.csv` | SKU, site, day | demand units |
| `inventory.csv` | SKU, site, as-of date | on hand, safety stock, on order |
| `purchase_orders.csv` | purchase order | supplier, quantity, planned arrival |
| `shipments.csv` | shipment status update | ETA, status, source quality |

Demand uses an over-dispersed gamma-Poisson approximation plus weekday
seasonality. The default seed injects a five-day delay from `SUP-A` for
`SKU-CRITICAL` into Pune-DC. This produces a controlled, known-ground-truth
success case. The failure scenario intentionally writes unverified or
conflicting shipment records in the test fixture.

The data is synthetic because openly available retail demand data rarely links
to purchase-order, shipment-status, and inventory facts at a common SKU/site
grain. Calibration to real historical demand patterns should precede a real
deployment.
