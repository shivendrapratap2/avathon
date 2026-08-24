"""Console tables must survive Arrow serialization.

Streamlit renders tables through pyarrow, which infers one type per column. A
column mixing int and str fails conversion and prints a traceback to the console
- harmless to the workflow, fatal to a live demo. These tests pin the contract.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from avathon.data_generation import generate
from avathon.display import proposal_rows, trace_rows
from avathon.llm import reference_planner_factory
from avathon.workflow import build_workflow


SKU, SITE = "SKU-CRITICAL", "Pune-DC"


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory) -> dict:
    data_dir = tmp_path_factory.mktemp("display") / "verified_delay"
    generate(data_dir, scenario="verified_delay")
    app = build_workflow(data_dir, planner_factory=reference_planner_factory(SKU, SITE))
    config = {"configurable": {"thread_id": "display"}}
    app.invoke(
        {
            "trace_id": "display", "scenario_id": "verified_delay", "sku": SKU,
            "site": SITE, "planner_id": "planner@example.com",
            "decision": "pending", "messages": [],
        },
        config=config,
    )
    return app.get_state(config).values


def _assert_arrow_convertible(rows: list[dict]) -> None:
    pa.Table.from_pandas(pd.DataFrame(rows))


def test_the_proposal_table_converts_to_arrow(completed_run: dict) -> None:
    _assert_arrow_convertible(proposal_rows(completed_run["proposal"]))


def test_the_trace_table_converts_to_arrow(completed_run: dict) -> None:
    _assert_arrow_convertible(trace_rows(completed_run["messages"]))


def test_a_mixed_type_column_would_have_been_caught(completed_run: dict) -> None:
    """The original defect, pinned: an int beside strings breaks conversion."""
    broken = proposal_rows(completed_run["proposal"])
    broken[1]["Value"] = completed_run["proposal"]["quantity"]  # a bare int
    with pytest.raises(pa.ArrowTypeError):
        _assert_arrow_convertible(broken)


def test_every_proposal_cell_is_a_string(completed_run: dict) -> None:
    for row in proposal_rows(completed_run["proposal"]):
        assert all(isinstance(value, str) for value in row.values()), row


def test_every_trace_cell_is_a_string(completed_run: dict) -> None:
    for row in trace_rows(completed_run["messages"]):
        assert all(isinstance(value, str) for value in row.values()), row


def test_a_monitor_proposal_renders_without_a_source_site(tmp_path: Path) -> None:
    """`source_site` is None whenever no stock moves; the table must not blank out."""
    data_dir = tmp_path / "healthy"
    generate(data_dir, scenario="healthy_stock")
    app = build_workflow(data_dir, planner_factory=reference_planner_factory(SKU, SITE))
    config = {"configurable": {"thread_id": "display-monitor"}}
    app.invoke(
        {
            "trace_id": "d2", "scenario_id": "healthy_stock", "sku": SKU, "site": SITE,
            "planner_id": "planner@example.com", "decision": "pending", "messages": [],
        },
        config=config,
    )
    proposal = app.get_state(config).values["proposal"]
    assert proposal["source_site"] is None

    rows = proposal_rows(proposal)
    _assert_arrow_convertible(rows)
    assert {"Field": "From", "Value": "—"} in rows
    assert {"Field": "Quantity", "Value": "0 units"} in rows


def test_an_empty_trace_still_renders(completed_run: dict) -> None:
    _assert_arrow_convertible(trace_rows([]) or [{"#": "", "From": ""}])
    assert trace_rows([]) == []
