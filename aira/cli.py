"""Command-line interface for AIRA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from aira import __version__
from aira.agent import run_agent_smoke, run_production_agent_smoke
from aira.benchmark import write_fixture_bundle, write_local_benchmark_bundle
from aira.bundles import validate_bundle
from aira.deepening import build_ara_deepening_plan, run_ara_deepening_experiment
from aira.manifest import DEFAULT_MANIFEST_PATH, load_manifest
from aira.memory import build_memory_index
from aira.migration import build_inventory
from aira.production_evaluation import evaluate_production_bundle
from aira.production_runner import run_production_experiment
from aira.registries import audit_registry, registry_payload


def _print_payload(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    status = payload.get("status") or ("valid" if payload.get("valid") else "invalid")
    print(f"{payload.get('schema_version', 'aira')}: {status}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AIRA AI research lab utilities.")
    parser.add_argument("--version", action="version", version=f"aira {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    labs = subparsers.add_parser("labs", help="Inspect the AIRA lab manifest.")
    labs_sub = labs.add_subparsers(dest="labs_command", required=True)
    labs_inspect = labs_sub.add_parser("inspect", help="Inspect research_lab.yaml.")
    labs_inspect.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH), help="Manifest path.")
    labs_inspect.add_argument("--json", action="store_true", help="Print JSON output.")

    bundles = subparsers.add_parser("bundles", help="Validate AIRA result bundles.")
    bundles_sub = bundles.add_subparsers(dest="bundles_command", required=True)
    bundles_validate = bundles_sub.add_parser("validate", help="Validate an aira_result_bundle.")
    bundles_validate.add_argument("path", help="Path to an AIRA result bundle directory.")
    bundles_validate.add_argument(
        "--profile",
        default="aira-mvp",
        choices=["aira-mvp", "ara-production", "ara-production-open"],
        help="Bundle validation profile.",
    )
    bundles_validate.add_argument("--json", action="store_true", help="Print JSON output.")

    migrate = subparsers.add_parser("migrate", help="Inventory legacy ARA AI experiment responsibilities.")
    migrate_sub = migrate.add_subparsers(dest="migrate_command", required=True)
    migrate_inventory = migrate_sub.add_parser("inventory", help="Build a read-only migration inventory.")
    migrate_inventory.add_argument("--source", required=True, help="Path to the legacy ARA platform repo.")
    migrate_inventory.add_argument("--json", action="store_true", help="Print JSON output.")

    benchmark = subparsers.add_parser("run-fixture-benchmark", help="Emit a deterministic fixture result bundle.")
    benchmark.add_argument("--out", required=True, help="Output bundle directory.")
    benchmark.add_argument("--json", action="store_true", help="Print JSON output.")

    local_benchmark = subparsers.add_parser(
        "run-local-benchmark",
        help="Run the deterministic local benchmark and emit an AIRA result bundle.",
    )
    local_benchmark.add_argument("--out", required=True, help="Output bundle directory.")
    local_benchmark.add_argument("--json", action="store_true", help="Print JSON output.")

    agent = subparsers.add_parser("agent", help="Run deterministic local experiment agent loops.")
    agent_sub = agent.add_subparsers(dest="agent_command", required=True)
    agent_smoke = agent_sub.add_parser("smoke", help="Run the deterministic local agent smoke.")
    agent_smoke.add_argument("--out", required=True, help="Output bundle directory.")
    agent_smoke.add_argument("--json", action="store_true", help="Print JSON output.")
    agent_production_smoke = agent_sub.add_parser(
        "production-smoke",
        help="Run the deterministic production-local ARA handoff smoke.",
    )
    agent_production_smoke.add_argument("--out", required=True, help="Output bundle directory.")
    agent_production_smoke.add_argument("--json", action="store_true", help="Print JSON output.")

    memory = subparsers.add_parser("memory", help="Build local experiment memory indexes.")
    memory_sub = memory.add_subparsers(dest="memory_command", required=True)
    memory_index = memory_sub.add_parser("index", help="Build a cross-run experiment memory index.")
    memory_index.add_argument(
        "--runs",
        required=True,
        action="append",
        help="AIRA result bundle directory or parent directory to scan. May be repeated.",
    )
    memory_index.add_argument("--out", required=True, help="Output memory index directory.")
    memory_index.add_argument(
        "--status",
        choices=["all", "passed", "failed", "unknown"],
        default="all",
        help="Lifecycle filter for runs promoted into the index.",
    )
    memory_index.add_argument(
        "--max-runs",
        type=int,
        help="Retain only the latest N matching runs in the emitted index.",
    )
    memory_index.add_argument(
        "--keep-existing",
        action="store_true",
        help="Keep the output directory and overwrite index files in place instead of rebuilding it.",
    )
    memory_index.add_argument("--json", action="store_true", help="Print JSON output.")

    experiments = subparsers.add_parser("experiments", help="Run bounded AIRA experiment plans.")
    experiments_sub = experiments.add_subparsers(dest="experiments_command", required=True)
    experiments_run = experiments_sub.add_parser("run", help="Run a policy-gated production experiment plan.")
    experiments_run.add_argument("--profile", required=True, help="Experiment execution profile.")
    experiments_run.add_argument("--plan", required=True, help="Production plan JSON path.")
    experiments_run.add_argument("--out", required=True, help="Output bundle directory.")
    experiments_run.add_argument("--json", action="store_true", help="Print JSON output.")
    experiments_deepen = experiments_sub.add_parser(
        "deepen",
        help="Generate and run an AIRA follow-up bundle from an ARA research-deepening task.",
    )
    experiments_deepen.add_argument("--profile", required=True, help="Experiment execution profile.")
    experiments_deepen.add_argument("--task", required=True, help="ARA research-deepening task package JSON path.")
    experiments_deepen.add_argument("--source-bundle", help="Optional source AIRA result bundle being deepened.")
    experiments_deepen.add_argument("--out", required=True, help="Output bundle directory.")
    experiments_deepen.add_argument("--json", action="store_true", help="Print JSON output.")
    experiments_plan = experiments_sub.add_parser(
        "build-deepening-plan",
        help="Build the production plan that would be used for an ARA deepening task.",
    )
    experiments_plan.add_argument("--profile", default="production-open", help="Experiment execution profile.")
    experiments_plan.add_argument("--task", required=True, help="ARA research-deepening task package JSON path.")
    experiments_plan.add_argument("--source-bundle", help="Optional source AIRA result bundle being deepened.")
    experiments_plan.add_argument("--out", help="Optional output plan JSON path.")
    experiments_plan.add_argument("--json", action="store_true", help="Print JSON output.")
    experiments_evaluate = experiments_sub.add_parser(
        "evaluate",
        help="Evaluate a production-local experiment bundle and append report artifacts.",
    )
    experiments_evaluate.add_argument("--bundle", required=True, help="AIRA production-local result bundle directory.")
    experiments_evaluate.add_argument("--json", action="store_true", help="Print JSON output.")

    registry = subparsers.add_parser("registry", help="Audit AIRA registry profiles.")
    registry_sub = registry.add_subparsers(dest="registry_command", required=True)
    registry_audit = registry_sub.add_parser("audit", help="Audit a registry profile.")
    registry_audit.add_argument("--profile", required=True, help="Registry profile to audit.")
    registry_audit.add_argument("--json", action="store_true", help="Print JSON output.")

    registries = subparsers.add_parser("registries", help="Print registry placeholders.")
    registries.add_argument("--profile", help="Optional explicit registry profile.")
    registries.add_argument("--json", action="store_true", help="Print JSON output.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "labs" and args.labs_command == "inspect":
        manifest = load_manifest(args.manifest)
        payload = manifest.to_dict()
        payload["status"] = "found" if manifest.validation.valid else "invalid"
        _print_payload(payload, as_json=args.json)
        return 0 if manifest.validation.valid else 1

    if args.command == "bundles" and args.bundles_command == "validate":
        result = validate_bundle(args.path, profile=args.profile)
        payload = result.to_dict()
        payload["status"] = "passed" if result.valid else "failed"
        _print_payload(payload, as_json=args.json)
        return 0 if result.valid else 1

    if args.command == "migrate" and args.migrate_command == "inventory":
        payload = build_inventory(args.source)
        payload["status"] = "passed" if payload["source_exists"] else "missing"
        _print_payload(payload, as_json=args.json)
        return 0 if payload["source_exists"] else 1

    if args.command == "run-fixture-benchmark":
        payload = write_fixture_bundle(Path(args.out))
        _print_payload(payload, as_json=args.json)
        return 0 if payload["status"] == "passed" else 1

    if args.command == "run-local-benchmark":
        payload = write_local_benchmark_bundle(Path(args.out))
        _print_payload(payload, as_json=args.json)
        return 0 if payload["status"] == "passed" else 1

    if args.command == "agent" and args.agent_command == "smoke":
        payload = run_agent_smoke(Path(args.out))
        _print_payload(payload, as_json=args.json)
        return 0 if payload["status"] == "passed" else 1

    if args.command == "agent" and args.agent_command == "production-smoke":
        payload = run_production_agent_smoke(Path(args.out))
        _print_payload(payload, as_json=args.json)
        return 0 if payload["status"] == "passed" else 1

    if args.command == "memory" and args.memory_command == "index":
        payload = build_memory_index(
            args.runs,
            args.out,
            status_filter=args.status,
            max_runs=args.max_runs,
            reset=not args.keep_existing,
        )
        _print_payload(payload, as_json=args.json)
        return 0 if payload["status"] == "passed" else 1

    if args.command == "experiments" and args.experiments_command == "run":
        payload = run_production_experiment(args.profile, args.plan, Path(args.out))
        _print_payload(payload, as_json=args.json)
        return 0 if payload["status"] == "passed" else 1

    if args.command == "experiments" and args.experiments_command == "deepen":
        payload = run_ara_deepening_experiment(
            profile_name=args.profile,
            task_package=args.task,
            source_bundle=args.source_bundle,
            output_dir=Path(args.out),
        )
        _print_payload(payload, as_json=args.json)
        return 0 if payload["status"] == "passed" else 1

    if args.command == "experiments" and args.experiments_command == "build-deepening-plan":
        payload = build_ara_deepening_plan(
            args.task,
            source_bundle=args.source_bundle,
            profile_name=args.profile,
        )
        if args.out:
            Path(args.out).expanduser().write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _print_payload(payload, as_json=args.json)
        return 0

    if args.command == "experiments" and args.experiments_command == "evaluate":
        payload = evaluate_production_bundle(Path(args.bundle))
        _print_payload(payload, as_json=args.json)
        return 0 if payload["status"] == "passed" else 1

    if args.command == "registry" and args.registry_command == "audit":
        payload = audit_registry(args.profile)
        _print_payload(payload, as_json=args.json)
        return 0 if payload["status"] == "passed" else 1

    if args.command == "registries":
        payload = registry_payload(args.profile)
        payload["status"] = "available"
        _print_payload(payload, as_json=args.json)
        return 0

    parser.error("Unhandled command.")
    return 2


def main_entry() -> int:
    return main(sys.argv[1:])
