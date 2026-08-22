"""Run success and failure demonstrations and write complete JSONL traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data_generation import generate
from .workflow import build_workflow


def run(scenario: str, project_root: Path) -> Path:
    data_dir = project_root / "data" / "generated"
    generate(data_dir, scenario="conflicting_evidence" if scenario == "failure" else "success")
    app = build_workflow(data_dir)
    config = {"configurable": {"thread_id": f"{scenario}-demo"}}
    initial = {
        "trace_id": f"trace-{scenario}-001", "scenario_id": scenario, "sku": "SKU-CRITICAL",
        "site": "Pune-DC", "decision": "pending", "messages": [],
    }
    app.invoke(initial, config=config)
    if scenario == "success":
        app.update_state(config, {"decision": "approve"})
        app.invoke(None, config=config)
    state = app.get_state(config).values
    output = project_root / "results" / f"trace_{scenario}.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for message in state["messages"]:
            handle.write(json.dumps(message, default=str) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["success", "failure"], required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(run(args.scenario, args.project_root))
