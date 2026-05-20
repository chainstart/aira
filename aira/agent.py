"""Deterministic local experiment agent loop for AIRA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aira.benchmark import (
    LOCAL_BENCHMARK_ID,
    LOCAL_COMMAND,
    LOCAL_DATASET_ID,
    LOCAL_MODEL_ID,
    LOCAL_TASK_ID,
    write_local_benchmark_bundle,
)
from aira.bundles import (
    ARA_GATE_PROFILE,
    ARA_HANDOFF_SCHEMA_VERSION,
    ARA_PRODUCTION_VALIDATION_PROFILE,
    BUNDLE_SCHEMA_VERSION,
    validate_bundle,
    write_json,
)
from aira.memory import build_memory_index
from aira.production_evaluation import evaluate_production_bundle
from aira.production_runner import PLAN_SCHEMA_VERSION, run_production_experiment
from aira.registries import registry_payload


AGENT_SCHEMA_VERSION = "aira.experiment_agent.v1"
AGENT_TASK_ID = "AIRA-AGENT-001"
AGENT_CREATED_AT = "2026-05-19T00:00:00Z"
AGENT_COMMAND = "python3 -m aira agent smoke"
PRODUCTION_AGENT_TASK_ID = "AIRA-PROD-ARA-001"
PRODUCTION_AGENT_COMMAND = "python3 -m aira agent production-smoke"
PRODUCTION_CREATED_AT = "2026-05-20T00:00:00Z"
ARA_PRODUCTION_VALIDATE_COMMAND = "python3 -m aira bundles validate <bundle> --profile ara-production --json"


def _by_id(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item.get("id")): item for item in items if isinstance(item.get("id"), str)}


def select_local_experiment(registry: dict[str, Any] | None = None) -> dict[str, Any]:
    """Select the bounded local benchmark the MVP agent is allowed to run."""
    payload = registry or registry_payload()
    datasets = _by_id(payload.get("datasets", []))
    models = _by_id(payload.get("models", []))
    for benchmark in payload.get("benchmarks", []):
        if benchmark.get("id") != LOCAL_BENCHMARK_ID:
            continue
        dataset_id = str(benchmark.get("dataset_id"))
        model_ids = [str(model_id) for model_id in benchmark.get("model_ids", [])]
        if dataset_id not in datasets or any(model_id not in models for model_id in model_ids):
            break
        if any(
            benchmark.get(flag) is True
            for flag in ("network_required", "external_datasets_required", "gpu_required", "live_model_calls")
        ):
            break
        return {
            "benchmark": benchmark,
            "dataset": datasets[dataset_id],
            "models": [models[model_id] for model_id in model_ids],
            "selection_reason": (
                "Selected the registered deterministic local benchmark because it emits an "
                "AIRA result bundle without network, GPU, external dataset, or live model requirements."
            ),
        }
    raise RuntimeError("No safe deterministic local benchmark is registered for the AIRA agent MVP.")


def build_agent_plan(output_dir: str | Path, registry: dict[str, Any] | None = None) -> dict[str, Any]:
    selection = select_local_experiment(registry)
    dataset = selection["dataset"]
    benchmark = selection["benchmark"]
    model_ids = [model["id"] for model in selection["models"]]
    return {
        "schema_version": "aira.agent_plan.v1",
        "agent_schema_version": AGENT_SCHEMA_VERSION,
        "task_id": AGENT_TASK_ID,
        "created_at": AGENT_CREATED_AT,
        "output_dir": str(Path(output_dir).expanduser().resolve()),
        "selected_registry_entries": {
            "benchmark_id": benchmark["id"],
            "dataset_id": dataset["id"],
            "model_ids": model_ids,
            "primary_model_id": LOCAL_MODEL_ID,
        },
        "selection_reason": selection["selection_reason"],
        "bounds": {
            "max_rows": dataset.get("rows"),
            "max_wall_time_seconds": 300,
            "network_required": False,
            "external_datasets_required": False,
            "gpu_required": False,
            "live_model_calls": False,
        },
        "steps": [
            {
                "phase": "plan",
                "status": "planned",
                "action": "Select a safe local benchmark from the AIRA registries.",
            },
            {
                "phase": "act",
                "status": "planned",
                "action": "Execute the registered deterministic local benchmark runner.",
                "command": LOCAL_COMMAND,
            },
            {
                "phase": "observe",
                "status": "planned",
                "action": "Validate the emitted AIRA result bundle and summarize metrics.",
            },
            {
                "phase": "reflect",
                "status": "planned",
                "action": "Persist reusable run memory for future local agent runs.",
            },
        ],
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _append_artifacts(bundle_path: Path, artifacts: list[dict[str, str]]) -> None:
    manifest_path = bundle_path / "artifact_manifest.json"
    manifest = _load_json(manifest_path)
    existing = {
        item.get("artifact_id")
        for item in manifest.get("artifacts", [])
        if isinstance(item, dict) and isinstance(item.get("artifact_id"), str)
    }
    for artifact in artifacts:
        if artifact["artifact_id"] not in existing:
            manifest.setdefault("artifacts", []).append(artifact)
    write_json(manifest_path, manifest)


def _production_smoke_plan() -> dict[str, Any]:
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_id": "production-local-fixture-plan",
        "description": "Deterministic production-local fixture plan for AIRA production ARA handoff smoke.",
        "network_required": False,
        "external_datasets_required": False,
        "gpu_required": False,
        "live_model_calls": False,
        "resource_requirements": {"python_packages": []},
        "tasks": [
            {
                "id": "prepare_dataset",
                "name": "Prepare deterministic local dataset",
                "dependencies": [],
                "command": {
                    "kind": "inline_python",
                    "code": "\n".join(
                        [
                            "import csv",
                            "from pathlib import Path",
                            "",
                            "rows = [",
                            "    {'example_id': 'prod-001', 'text': 'policy checks passed', 'label': 'pass'},",
                            "    {'example_id': 'prod-002', 'text': 'runner blocked unsafe package', 'label': 'fail'},",
                            "    {'example_id': 'prod-003', 'text': 'artifacts materialized locally', 'label': 'pass'},",
                            "    {'example_id': 'prod-004', 'text': 'dependency failed before output', 'label': 'fail'},",
                            "]",
                            "with Path('dataset.csv').open('w', newline='', encoding='utf-8') as handle:",
                            "    writer = csv.DictWriter(handle, fieldnames=['example_id', 'text', 'label'])",
                            "    writer.writeheader()",
                            "    writer.writerows(rows)",
                            "Path('results.json').write_text('{\"success\": true, \"row_count\": 4}\\n', encoding='utf-8')",
                            "print('prepared 4 rows')",
                        ]
                    ),
                },
                "outputs": [
                    {
                        "artifact_id": "production_dataset",
                        "path": "dataset.csv",
                        "kind": "dataset",
                        "description": "Deterministic production-local fixture dataset.",
                    },
                    {
                        "artifact_id": "prepare_results",
                        "path": "results.json",
                        "kind": "task_results",
                        "description": "Prepare task result marker.",
                    },
                ],
            },
            {
                "id": "score_dataset",
                "name": "Score deterministic local dataset",
                "dependencies": ["prepare_dataset"],
                "command": {
                    "kind": "inline_python",
                    "code": "\n".join(
                        [
                            "import csv",
                            "import json",
                            "import os",
                            "from pathlib import Path",
                            "",
                            "dep_dirs = json.loads(os.environ['AIRA_DEP_DIRS'])",
                            "dataset_path = Path(dep_dirs['prepare_dataset']) / 'dataset.csv'",
                            "rows = list(csv.DictReader(dataset_path.open(encoding='utf-8')))",
                            "correct = 0",
                            "predictions = []",
                            "for row in rows:",
                            "    prediction = 'fail' if any(term in row['text'] for term in ['blocked', 'failed']) else 'pass'",
                            "    correct += int(prediction == row['label'])",
                            "    predictions.append({**row, 'prediction': prediction})",
                            "accuracy = correct / len(rows)",
                            "Path('metrics.json').write_text(json.dumps({'success': True, 'accuracy': accuracy, 'row_count': len(rows)}, sort_keys=True) + '\\n', encoding='utf-8')",
                            "with Path('predictions.csv').open('w', newline='', encoding='utf-8') as handle:",
                            "    writer = csv.DictWriter(handle, fieldnames=['example_id', 'text', 'label', 'prediction'])",
                            "    writer.writeheader()",
                            "    writer.writerows(predictions)",
                            "print(json.dumps({'accuracy': accuracy, 'row_count': len(rows)}, sort_keys=True))",
                        ]
                    ),
                },
                "outputs": [
                    {
                        "artifact_id": "production_metrics",
                        "path": "metrics.json",
                        "kind": "metrics",
                        "description": "Deterministic production-local scoring metrics.",
                    },
                    {
                        "artifact_id": "production_predictions",
                        "path": "predictions.csv",
                        "kind": "predictions",
                        "description": "Deterministic per-example production-local predictions.",
                    },
                ],
            },
        ],
    }


def _append_agent_claim(bundle_path: Path) -> None:
    claims_path = bundle_path / "claims.json"
    claims_payload = _load_json(claims_path)
    claims = claims_payload.setdefault("claims", [])
    if any(isinstance(claim, dict) and claim.get("claim_id") == "aira-agent-smoke-c1" for claim in claims):
        write_json(claims_path, claims_payload)
        return
    claims.append(
        {
            "claim_id": "aira-agent-smoke-c1",
            "claim": (
                "The deterministic AIRA experiment agent selected a registered local benchmark, "
                "executed it, validated the result bundle, and persisted reusable run memory."
            ),
            "status": "confirmed",
            "reproduction_status": "reproduced",
            "supported_by": [
                "reproduction_status",
                "ara_handoff",
                "reproducibility_notes",
                "agent_plan",
                "agent_trace",
                "agent_observation",
                "agent_memory",
            ],
            "limitations": [
                "The MVP agent can execute only the registered deterministic local benchmark runner.",
                "Agent memory is persisted in the emitted result bundle, not in a shared service.",
                "No live model APIs, GPU execution, external datasets, or network access are used.",
            ],
        }
    )
    write_json(claims_path, claims_payload)


def _update_bundle_manifest(bundle_path: Path, plan: dict[str, Any]) -> None:
    manifest_path = bundle_path / "bundle_manifest.json"
    manifest = _load_json(manifest_path)
    if "task_id" in manifest and manifest["task_id"] != AGENT_TASK_ID:
        manifest["source_benchmark_task_id"] = manifest["task_id"]
    manifest["task_id"] = AGENT_TASK_ID
    manifest["agent"] = {
        "schema_version": AGENT_SCHEMA_VERSION,
        "task_id": AGENT_TASK_ID,
        "command": AGENT_COMMAND,
        "loop": "plan-act-observe-reflect",
        "selected_benchmark_id": plan["selected_registry_entries"]["benchmark_id"],
    }
    write_json(manifest_path, manifest)


def _update_ara_handoff(bundle_path: Path, plan: dict[str, Any], run_id: str) -> None:
    handoff_path = bundle_path / "artifacts" / "ara_handoff.json"
    if not handoff_path.exists():
        return
    handoff = _load_json(handoff_path)
    handoff["task_id"] = AGENT_TASK_ID
    handoff["source_benchmark_task_id"] = LOCAL_TASK_ID
    handoff["reproduce_command"] = f"{AGENT_COMMAND} --out <bundle>"
    handoff.setdefault("dispatch", {})["entrypoint"] = AGENT_COMMAND
    handoff["agent_smoke"] = {
        "schema_version": "aira.ara_agent_handoff.v1",
        "agent_schema_version": AGENT_SCHEMA_VERSION,
        "task_id": AGENT_TASK_ID,
        "run_id": run_id,
        "loop": "plan-act-observe-reflect",
        "selected_registry_entries": plan["selected_registry_entries"],
        "artifacts": [
            "artifacts/agent_plan.json",
            "artifacts/agent_trace.json",
            "artifacts/agent_observation.json",
            "artifacts/agent_reflection.json",
            "memory/agent_memory.json",
            "memory/agent_memory.jsonl",
        ],
    }
    required_inputs = handoff.setdefault("required_gate_inputs", {})
    required_inputs.update(
        {
            "agent_plan": "artifacts/agent_plan.json",
            "agent_trace": "artifacts/agent_trace.json",
            "agent_observation": "artifacts/agent_observation.json",
            "agent_reflection": "artifacts/agent_reflection.json",
            "agent_memory": "memory/agent_memory.json",
        }
    )
    write_json(handoff_path, handoff)

    notes_path = bundle_path / "artifacts" / "reproducibility_notes.md"
    if notes_path.exists():
        notes = notes_path.read_text(encoding="utf-8").rstrip()
        notes_path.write_text(
            "\n".join(
                [
                    notes,
                    "",
                    "## Agent Smoke Handoff",
                    "",
                    f"- Agent task id: `{AGENT_TASK_ID}`.",
                    f"- Reproduce the agent smoke with `{AGENT_COMMAND} --out <bundle>`.",
                    "- The agent loop only selects the registered deterministic local benchmark.",
                    "- The emitted bundle includes agent plan, trace, observation, reflection, ablation, error-analysis, and memory artifacts.",
                    "",
                ]
            ),
            encoding="utf-8",
        )


def _load_production_fingerprints(bundle_path: Path) -> dict[str, str]:
    provenance = _load_json(bundle_path / "artifacts" / "provenance.json")
    fingerprints = provenance.get("input_fingerprints")
    return fingerprints if isinstance(fingerprints, dict) else {}


def _write_production_reproducibility_notes(bundle_path: Path, run_id: str) -> None:
    fingerprints = _load_production_fingerprints(bundle_path)
    (bundle_path / "artifacts" / "reproducibility_notes.md").write_text(
        "\n".join(
            [
                "# Production ARA Reproducibility Notes",
                "",
                f"- Run id: `{run_id}`.",
                f"- Reproduce the production smoke with `{PRODUCTION_AGENT_COMMAND} --out <bundle> --json`.",
                f"- Validate the ARA production handoff with `{ARA_PRODUCTION_VALIDATE_COMMAND}`.",
                "- ARA dispatch is intentionally limited to `research_lab.yaml` and `aira_result_bundle`.",
                "- The runner profile is `production-local`; package installation, network, GPU, external datasets, and live model calls are disabled.",
                f"- Dataset sha256: `{fingerprints.get('dataset_sha256', '')}`.",
                f"- Model config sha256: `{fingerprints.get('model_config_sha256', '')}`.",
                f"- Registry snapshot sha256: `{fingerprints.get('registry_snapshot_sha256', '')}`.",
                "- The bundle includes production policy, execution trace, evaluation reports, run ledger, and local memory index artifacts.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_production_ara_handoff(bundle_path: Path, run_id: str) -> dict[str, Any]:
    fingerprints = _load_production_fingerprints(bundle_path)
    handoff = {
        "schema_version": ARA_HANDOFF_SCHEMA_VERSION,
        "consumer": "ara",
        "gate_profile": ARA_GATE_PROFILE,
        "validation_profile": ARA_PRODUCTION_VALIDATION_PROFILE,
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "bundle_type": "aira_result_bundle",
        "producer": "aira",
        "task_id": PRODUCTION_AGENT_TASK_ID,
        "run_id": run_id,
        "created_at": PRODUCTION_CREATED_AT,
        "status": "ready",
        "validation_command": ARA_PRODUCTION_VALIDATE_COMMAND,
        "reproduce_command": f"{PRODUCTION_AGENT_COMMAND} --out <bundle> --json",
        "required_gate_inputs": {
            "bundle_manifest": "bundle_manifest.json",
            "artifact_manifest": "artifact_manifest.json",
            "claims": "claims.json",
            "writing_brief": "writing_brief.md",
            "limitations": "limitations.md",
            "reproducibility_notes": "artifacts/reproducibility_notes.md",
            "reproduction_status": "artifacts/reproduction_status.json",
            "provenance": "artifacts/provenance.json",
            "run_ledger_entry": "artifacts/run_ledger_entry.json",
            "run_ledger": "memory/run_ledger.jsonl",
            "production_plan": "artifacts/production_plan.json",
            "policy_report": "artifacts/policy_report.json",
            "execution_trace": "artifacts/execution_trace.json",
            "task_summary": "artifacts/task_summary.json",
            "production_evaluation_metrics": "artifacts/production_evaluation/metrics.json",
            "production_ablation_matrix": "artifacts/production_evaluation/ablation_matrix.json",
            "production_error_taxonomy": "artifacts/production_evaluation/error_taxonomy.json",
            "production_statistical_tests": "artifacts/production_evaluation/statistical_tests.json",
            "production_report_summary": "artifacts/production_evaluation/report_summary.json",
            "memory_index": "memory/production_index/memory_index.json",
            "memory_runs": "memory/production_index/runs.jsonl",
            "memory_failures": "memory/production_index/failures.jsonl",
            "memory_fingerprints": "memory/production_index/fingerprints.json",
            "memory_outcomes": "memory/production_index/outcomes.json",
            "memory_reflections": "memory/production_index/reflections.json",
        },
        "reproducibility": {
            "deterministic": True,
            "network_required": False,
            "external_datasets_required": False,
            "gpu_required": False,
            "live_model_calls": False,
            "input_fingerprints": fingerprints,
        },
        "claim_gate": {
            "confirmed_claims_require_reproduced_status": True,
            "confirmed_claims_require_reproduction_artifact": True,
        },
        "dispatch": {
            "lab_id": "aira",
            "manifest_path": "research_lab.yaml",
            "bundle_type": "aira_result_bundle",
            "profile": "production-local",
            "validation_profile": ARA_PRODUCTION_VALIDATION_PROFILE,
            "entrypoint": PRODUCTION_AGENT_COMMAND,
            "runner_entrypoint": "python3 -m aira experiments run --profile production-local",
            "evaluation_entrypoint": "python3 -m aira experiments evaluate",
            "validation_command": ARA_PRODUCTION_VALIDATE_COMMAND,
            "allowed_interfaces": ["research_lab.yaml", "aira_result_bundle"],
            "side_effect_free_validation": True,
            "network_policy": "none",
        },
        "production_local": {
            "schema_version": "aira.ara_production_handoff.v1",
            "runner_profile": "production-local",
            "plan_artifact": "artifacts/production_plan.json",
            "policy_artifact": "artifacts/policy_report.json",
            "trace_artifact": "artifacts/execution_trace.json",
            "evaluation_artifacts": [
                "artifacts/production_evaluation/metrics.json",
                "artifacts/production_evaluation/ablation_matrix.json",
                "artifacts/production_evaluation/error_taxonomy.json",
                "artifacts/production_evaluation/statistical_tests.json",
                "artifacts/production_evaluation/report_summary.json",
            ],
            "memory_index_artifacts": [
                "memory/production_index/memory_index.json",
                "memory/production_index/runs.jsonl",
                "memory/production_index/failures.jsonl",
                "memory/production_index/fingerprints.json",
                "memory/production_index/outcomes.json",
                "memory/production_index/reflections.json",
            ],
        },
    }
    write_json(bundle_path / "artifacts" / "ara_handoff.json", handoff)
    return handoff


def _append_production_handoff_claim(bundle_path: Path) -> None:
    claims_path = bundle_path / "claims.json"
    payload = _load_json(claims_path)
    claim = {
        "claim_id": "aira-production-ara-handoff-c1",
        "claim": (
            "The AIRA production-local output exposes ARA dispatch and validation through "
            "research_lab.yaml metadata and the aira_result_bundle contract only."
        ),
        "status": "confirmed",
        "reproduction_status": "reproduced",
        "supported_by": [
            "reproduction_status",
            "ara_handoff",
            "reproducibility_notes",
            "production_memory_index",
            "production_evaluation_metrics",
            "production_report_summary",
        ],
        "limitations": [
            "The profile is production-local and does not enable network, GPU, external datasets, package installation, or live model calls.",
            "The smoke plan is deterministic and local; broader production plans must supply their own bundle artifacts.",
            "ARA consumes the manifest and bundle contract, not AIRA-internal Python modules.",
        ],
    }
    claims = payload.setdefault("claims", [])
    payload["claims"] = [item for item in claims if item.get("claim_id") != claim["claim_id"]] + [claim]
    write_json(claims_path, payload)


def _update_production_bundle_manifest(bundle_path: Path) -> None:
    manifest_path = bundle_path / "bundle_manifest.json"
    manifest = _load_json(manifest_path)
    manifest["task_id"] = PRODUCTION_AGENT_TASK_ID
    manifest["ara_handoff"] = {
        "schema_version": ARA_HANDOFF_SCHEMA_VERSION,
        "gate_profile": ARA_GATE_PROFILE,
        "validation_profile": ARA_PRODUCTION_VALIDATION_PROFILE,
        "artifact": "artifacts/ara_handoff.json",
        "validation_command": ARA_PRODUCTION_VALIDATE_COMMAND,
        "dispatch_manifest": "research_lab.yaml",
        "allowed_interfaces": ["research_lab.yaml", "aira_result_bundle"],
    }
    manifest["production_agent"] = {
        "schema_version": "aira.production_agent_smoke.v1",
        "task_id": PRODUCTION_AGENT_TASK_ID,
        "command": PRODUCTION_AGENT_COMMAND,
        "profile": "production-local",
        "validation_profile": ARA_PRODUCTION_VALIDATION_PROFILE,
    }
    write_json(manifest_path, manifest)


def _append_production_handoff_artifacts(bundle_path: Path) -> None:
    _append_artifacts(
        bundle_path,
        [
            {
                "artifact_id": "ara_handoff",
                "path": "artifacts/ara_handoff.json",
                "kind": "ara_handoff",
                "description": "ARA production-local handoff dispatch and gate metadata.",
            },
            {
                "artifact_id": "reproducibility_notes",
                "path": "artifacts/reproducibility_notes.md",
                "kind": "reproducibility_notes",
                "description": "Production-local ARA reproducibility and validation notes.",
            },
            {
                "artifact_id": "production_memory_index",
                "path": "memory/production_index/memory_index.json",
                "kind": "production_memory_index",
                "description": "Local cross-run memory index for the production handoff run.",
            },
            {
                "artifact_id": "production_memory_runs",
                "path": "memory/production_index/runs.jsonl",
                "kind": "production_memory_runs",
                "description": "Run summaries promoted into the production memory index.",
            },
            {
                "artifact_id": "production_memory_failures",
                "path": "memory/production_index/failures.jsonl",
                "kind": "production_memory_failures",
                "description": "Failure ledger promoted into the production memory index.",
            },
            {
                "artifact_id": "production_memory_fingerprints",
                "path": "memory/production_index/fingerprints.json",
                "kind": "production_memory_fingerprints",
                "description": "Fingerprint lookup table for the production memory index.",
            },
            {
                "artifact_id": "production_memory_outcomes",
                "path": "memory/production_index/outcomes.json",
                "kind": "production_memory_outcomes",
                "description": "Dataset and model outcome matrix for the production memory index.",
            },
            {
                "artifact_id": "production_memory_reflections",
                "path": "memory/production_index/reflections.json",
                "kind": "production_memory_reflections",
                "description": "Agent reflection retrieval table for the production memory index.",
            },
        ],
    )


def run_agent_smoke(output_dir: str | Path) -> dict[str, Any]:
    """Run the local deterministic agent MVP and emit an updated result bundle."""
    out = Path(output_dir).expanduser().resolve()
    plan = build_agent_plan(out)
    benchmark_payload = write_local_benchmark_bundle(out)
    initial_validation = validate_bundle(out)
    artifact_manifest = _load_json(out / "artifact_manifest.json")
    artifact_ids = [
        artifact["artifact_id"]
        for artifact in artifact_manifest.get("artifacts", [])
        if isinstance(artifact, dict) and isinstance(artifact.get("artifact_id"), str)
    ]
    observation = {
        "schema_version": "aira.agent_observation.v1",
        "task_id": AGENT_TASK_ID,
        "created_at": AGENT_CREATED_AT,
        "bundle_path": str(out),
        "bundle_valid": initial_validation.valid,
        "validation_error_count": len(initial_validation.errors),
        "artifact_ids": sorted(artifact_ids),
        "metrics": benchmark_payload["benchmark"]["metrics"],
        "analysis": benchmark_payload["analysis"],
        "run_id": benchmark_payload["run_id"],
    }
    reflection = {
        "schema_version": "aira.agent_reflection.v1",
        "task_id": AGENT_TASK_ID,
        "created_at": AGENT_CREATED_AT,
        "run_id": benchmark_payload["run_id"],
        "outcome": "accepted" if initial_validation.valid else "rejected",
        "reusable_memory": [
            "Use local-text-outcome-classification for offline smoke checks.",
            "Preserve negative outcome terms; the deterministic ablation records fail-example collapse without them.",
            "Carry experiment_memory artifacts forward when comparing local agent runs.",
            "Require bundle validation before promoting any agent-produced result.",
            "Keep live_model_calls, network_required, external_datasets_required, and gpu_required false.",
        ],
        "next_actions": [
            "Add more deterministic local runners before enabling agent choice among experiment families.",
            "Promote bundle-local experiment memory to a shared local memory index when cross-run retrieval is needed.",
        ],
    }
    memory_entry = {
        "schema_version": "aira.agent_memory_entry.v1",
        "agent_schema_version": AGENT_SCHEMA_VERSION,
        "task_id": AGENT_TASK_ID,
        "created_at": AGENT_CREATED_AT,
        "run_id": benchmark_payload["run_id"],
        "bundle_path": str(out),
        "selected_registry_entries": plan["selected_registry_entries"],
        "metrics": benchmark_payload["benchmark"]["metrics"],
        "analysis": benchmark_payload["analysis"],
        "experiment_memory": {
            "path": benchmark_payload["experiment_memory"]["path"],
            "retrieval_keys": benchmark_payload["experiment_memory"]["entry"]["retrieval_keys"],
            "ablation_findings": benchmark_payload["experiment_memory"]["entry"]["ablation_findings"],
        },
        "bundle_valid": initial_validation.valid,
        "outcome": reflection["outcome"],
        "reusable_notes": reflection["reusable_memory"],
    }
    trace = {
        "schema_version": "aira.agent_trace.v1",
        "agent_schema_version": AGENT_SCHEMA_VERSION,
        "task_id": AGENT_TASK_ID,
        "created_at": AGENT_CREATED_AT,
        "loop": [
            {"phase": "plan", "status": "completed", "artifact": "artifacts/agent_plan.json"},
            {
                "phase": "act",
                "status": "completed" if benchmark_payload["status"] == "passed" else "failed",
                "command": LOCAL_COMMAND,
                "run_id": benchmark_payload["run_id"],
            },
            {
                "phase": "observe",
                "status": "completed" if initial_validation.valid else "failed",
                "artifact": "artifacts/agent_observation.json",
            },
            {
                "phase": "reflect",
                "status": "completed",
                "artifact": "artifacts/agent_reflection.json",
            },
        ],
        "plan": plan,
        "observation": observation,
        "reflection": reflection,
        "memory_entry": memory_entry,
    }

    write_json(out / "artifacts" / "agent_plan.json", plan)
    write_json(out / "artifacts" / "agent_observation.json", observation)
    write_json(out / "artifacts" / "agent_reflection.json", reflection)
    write_json(out / "artifacts" / "agent_trace.json", trace)
    write_json(
        out / "memory" / "agent_memory.json",
        {"schema_version": "aira.agent_memory.v1", "entries": [memory_entry]},
    )
    (out / "memory" / "agent_memory.jsonl").write_text(
        json.dumps(memory_entry, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _append_artifacts(
        out,
        [
            {
                "artifact_id": "agent_plan",
                "path": "artifacts/agent_plan.json",
                "kind": "agent_plan",
                "description": "Deterministic AIRA agent plan over local registries.",
            },
            {
                "artifact_id": "agent_observation",
                "path": "artifacts/agent_observation.json",
                "kind": "agent_observation",
                "description": "AIRA agent observation after bounded local benchmark execution.",
            },
            {
                "artifact_id": "agent_reflection",
                "path": "artifacts/agent_reflection.json",
                "kind": "agent_reflection",
                "description": "AIRA agent reflection and reusable lessons.",
            },
            {
                "artifact_id": "agent_trace",
                "path": "artifacts/agent_trace.json",
                "kind": "agent_trace",
                "description": "Plan-act-observe-reflect trace for the local experiment agent.",
            },
            {
                "artifact_id": "agent_memory",
                "path": "memory/agent_memory.json",
                "kind": "agent_memory",
                "description": "Reusable bundle-local memory entries emitted by the AIRA agent.",
            },
            {
                "artifact_id": "agent_memory_log",
                "path": "memory/agent_memory.jsonl",
                "kind": "agent_memory",
                "description": "JSONL form of reusable AIRA agent memory.",
            },
        ],
    )
    _append_agent_claim(out)
    _update_bundle_manifest(out, plan)
    _update_ara_handoff(out, plan, benchmark_payload["run_id"])
    final_validation = validate_bundle(out)
    return {
        "schema_version": "aira.agent_smoke.v1",
        "status": "passed" if final_validation.valid and reflection["outcome"] == "accepted" else "failed",
        "bundle_path": str(out),
        "run_id": benchmark_payload["run_id"],
        "selected_registry_entries": plan["selected_registry_entries"],
        "loop": trace["loop"],
        "plan": plan,
        "observation": observation,
        "reflection": reflection,
        "memory": {
            "path": str(out / "memory" / "agent_memory.json"),
            "entry": memory_entry,
        },
        "validation": final_validation.to_dict(),
    }


def run_production_agent_smoke(output_dir: str | Path) -> dict[str, Any]:
    """Run a deterministic production-local handoff smoke and emit an ARA-ready bundle."""
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "inputs" / "production_smoke_plan.json"
    write_json(plan_path, _production_smoke_plan())

    runner_payload = run_production_experiment("production-local", plan_path, out)
    evaluation_payload = evaluate_production_bundle(out)
    memory_payload = build_memory_index([out], out / "memory" / "production_index")
    _write_production_reproducibility_notes(out, runner_payload["run_id"])
    handoff = _write_production_ara_handoff(out, runner_payload["run_id"])
    _append_production_handoff_artifacts(out)
    _append_production_handoff_claim(out)
    _update_production_bundle_manifest(out)

    final_validation = validate_bundle(out, profile=ARA_PRODUCTION_VALIDATION_PROFILE)
    status = "passed" if runner_payload["status"] == "passed" and evaluation_payload["status"] == "passed" and final_validation.valid else "failed"
    return {
        "schema_version": "aira.production_agent_smoke.v1",
        "status": status,
        "task_id": PRODUCTION_AGENT_TASK_ID,
        "profile": "production-local",
        "validation_profile": ARA_PRODUCTION_VALIDATION_PROFILE,
        "bundle_path": str(out),
        "run_id": runner_payload["run_id"],
        "runner": runner_payload,
        "evaluation": evaluation_payload,
        "memory_index": {
            "path": str(out / "memory" / "production_index"),
            "run_count": memory_payload["run_count"],
            "failure_count": memory_payload["failure_count"],
            "reflection_count": memory_payload["reflection_count"],
        },
        "handoff": handoff,
        "validation": final_validation.to_dict(),
    }
