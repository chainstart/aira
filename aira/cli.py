"""Command-line interface for AIRA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from aira import __version__
from aira.benchmark import write_fixture_bundle
from aira.bundles import validate_bundle
from aira.manifest import DEFAULT_MANIFEST_PATH, load_manifest
from aira.migration import build_inventory
from aira.registries import registry_payload


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
    bundles_validate.add_argument("--json", action="store_true", help="Print JSON output.")

    migrate = subparsers.add_parser("migrate", help="Inventory legacy ARA AI experiment responsibilities.")
    migrate_sub = migrate.add_subparsers(dest="migrate_command", required=True)
    migrate_inventory = migrate_sub.add_parser("inventory", help="Build a read-only migration inventory.")
    migrate_inventory.add_argument("--source", required=True, help="Path to the legacy auto-research-agent repo.")
    migrate_inventory.add_argument("--json", action="store_true", help="Print JSON output.")

    benchmark = subparsers.add_parser("run-fixture-benchmark", help="Emit a deterministic fixture result bundle.")
    benchmark.add_argument("--out", required=True, help="Output bundle directory.")
    benchmark.add_argument("--json", action="store_true", help="Print JSON output.")

    registries = subparsers.add_parser("registries", help="Print registry placeholders.")
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
        result = validate_bundle(args.path)
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

    if args.command == "registries":
        payload = registry_payload()
        payload["status"] = "available"
        _print_payload(payload, as_json=args.json)
        return 0

    parser.error("Unhandled command.")
    return 2


def main_entry() -> int:
    return main(sys.argv[1:])
