"""Run one scenario end to end and write a complete JSONL trace."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .data_generation import SCENARIOS, generate
from .llm import DEFAULT_MODEL, reference_planner_factory
from .workflow import build_workflow


def run(
    scenario: str,
    project_root: Path,
    *,
    sku: str = "SKU-CRITICAL",
    site: str = "Pune-DC",
    decision: str = "approve",
    planner_id: str = "planner.demo@example.com",
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
) -> Path:
    """Generate the scenario, drive the graph, and persist the trace."""
    data_dir = project_root / "data" / "generated" / scenario
    generate(data_dir, scenario=scenario)

    if api_key:
        app = build_workflow(data_dir, api_key=api_key, model=model)
    else:
        app = build_workflow(data_dir, planner_factory=reference_planner_factory(sku, site))

    config = {"configurable": {"thread_id": f"{scenario}-demo"}}
    app.invoke(
        {
            "trace_id": f"trace-{scenario}", "scenario_id": scenario,
            "sku": sku, "site": site, "planner_id": planner_id,
            "decision": "pending", "messages": [],
        },
        config=config,
    )

    # Resume only if the graph actually paused at the human gate.
    if app.get_state(config).next:
        app.update_state(config, {"decision": decision})
        app.invoke(None, config=config)

    state = app.get_state(config).values
    output_dir = project_root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"trace_{scenario}.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for message in state["messages"]:
            handle.write(json.dumps(message, default=str) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--decision", choices=["approve", "reject"], default="approve")
    parser.add_argument(
        "--live", action="store_true",
        help="Plan with the OpenAI model in OPENAI_API_KEY instead of the reference plan.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    print(
        run(
            args.scenario, args.project_root, decision=args.decision,
            api_key=os.environ.get("OPENAI_API_KEY") if args.live else None,
            model=args.model,
        )
    )
