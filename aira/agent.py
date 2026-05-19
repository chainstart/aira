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
    write_local_benchmark_bundle,
)
from aira.bundles import validate_bundle, write_json
from aira.registries import registry_payload


AGENT_SCHEMA_VERSION = "aira.experiment_agent.v1"
AGENT_TASK_ID = "AIRA-AGENT-001"
AGENT_CREATED_AT = "2026-05-19T00:00:00Z"
AGENT_COMMAND = "python3 -m aira agent smoke"


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
            "Require bundle validation before promoting any agent-produced result.",
            "Keep live_model_calls, network_required, external_datasets_required, and gpu_required false.",
        ],
        "next_actions": [
            "Add more deterministic local runners before enabling agent choice among experiment families.",
            "Promote bundle-local memory to a shared local memory index when cross-run retrieval is needed.",
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
