"""Local cross-run experiment memory index for AIRA bundles."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from aira.bundles import write_json


MEMORY_INDEX_SCHEMA_VERSION = "aira.memory_index.v1"
MEMORY_TASK_ID = "AIRA-PROD-MEMORY-001"


def _created_at() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} line {index} must contain a JSON object.")
        rows.append(payload)
    return rows


def _bundle_dirs(roots: Iterable[str | Path]) -> list[Path]:
    bundles: set[Path] = set()
    for root_value in roots:
        root = Path(root_value).expanduser().resolve()
        if (root / "bundle_manifest.json").exists():
            bundles.add(root)
            continue
        if root.exists():
            for manifest in root.rglob("bundle_manifest.json"):
                bundles.add(manifest.parent)
    return sorted(bundles, key=lambda item: str(item))


def _ledger_entries(bundle: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in [
        _read_json_if_exists(bundle / "artifacts" / "run_ledger_entry.json"),
        *_read_jsonl(bundle / "memory" / "run_ledger.jsonl"),
    ]:
        if not isinstance(candidate, dict):
            continue
        key = json.dumps(
            {
                "run_id": candidate.get("run_id"),
                "status": candidate.get("status"),
                "benchmark_id": candidate.get("benchmark_id"),
            },
            sort_keys=True,
        )
        if key in seen:
            continue
        seen.add(key)
        entries.append(candidate)
    if entries:
        return entries
    return [
        {
            "schema_version": "aira.run_ledger_entry.v1",
            "run_id": manifest.get("run_id", bundle.name),
            "task_id": manifest.get("task_id"),
            "created_at": manifest.get("created_at"),
            "status": "unknown",
            "bundle_path": str(bundle),
            "bundle_type": manifest.get("bundle_type"),
            "benchmark_id": manifest.get("benchmark_id"),
            "dataset_id": manifest.get("dataset_id"),
            "model_id": manifest.get("model_id"),
            "metrics": {},
            "provenance": {},
            "reproducibility": {},
            "artifacts": [],
        }
    ]


def _experiment_memory_for_run(bundle: Path, run_id: str) -> dict[str, Any] | None:
    candidates = []
    direct = _read_json_if_exists(bundle / "memory" / "experiment_memory.json")
    if direct:
        candidates.append(direct)
    candidates.extend(_read_jsonl(bundle / "memory" / "experiment_memory.jsonl"))
    for item in candidates:
        if item.get("run_id") == run_id:
            return item
    return candidates[0] if candidates else None


def _agent_memory_for_run(bundle: Path, run_id: str) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    payload = _read_json_if_exists(bundle / "memory" / "agent_memory.json")
    if payload:
        entries = payload.get("entries", [])
        candidates.extend(item for item in entries if isinstance(item, dict))
    candidates.extend(_read_jsonl(bundle / "memory" / "agent_memory.jsonl"))
    for item in candidates:
        if item.get("run_id") == run_id:
            return item
    return candidates[0] if candidates else None


def _input_fingerprints(
    bundle: Path,
    manifest: dict[str, Any],
    ledger: dict[str, Any],
    experiment_memory: dict[str, Any] | None,
) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    provenance = _read_json_if_exists(bundle / "artifacts" / "provenance.json") or {}
    for source in [
        provenance.get("input_fingerprints"),
        experiment_memory.get("input_fingerprints") if experiment_memory else None,
        ledger.get("provenance"),
        manifest.get("input_fingerprints"),
    ]:
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            if isinstance(value, str) and len(value) == 64:
                fingerprints[key] = value
    return fingerprints


def _evaluation_summary(bundle: Path) -> dict[str, Any]:
    metrics = _read_json_if_exists(bundle / "artifacts" / "production_evaluation" / "metrics.json")
    ablation = _read_json_if_exists(bundle / "artifacts" / "production_evaluation" / "ablation_matrix.json")
    taxonomy = _read_json_if_exists(bundle / "artifacts" / "production_evaluation" / "error_taxonomy.json")
    report = _read_json_if_exists(bundle / "artifacts" / "production_evaluation" / "report_summary.json")
    if not any([metrics, ablation, taxonomy, report]):
        return {}
    ablation_rows = ablation.get("rows", []) if isinstance(ablation, dict) else []
    return {
        "metrics": metrics.get("metrics", {}) if isinstance(metrics, dict) else {},
        "row_count": metrics.get("row_count") if isinstance(metrics, dict) else None,
        "primary_error_count": taxonomy.get("primary_error_count", 0) if isinstance(taxonomy, dict) else 0,
        "ablation_error_count": taxonomy.get("ablation_error_count", 0) if isinstance(taxonomy, dict) else 0,
        "taxonomy": taxonomy.get("taxonomy", []) if isinstance(taxonomy, dict) else [],
        "ablation_rows": ablation_rows if isinstance(ablation_rows, list) else [],
        "recommendations": report.get("recommendations", []) if isinstance(report, dict) else [],
    }


def _agent_reflection(bundle: Path, run_id: str, agent_memory: dict[str, Any] | None) -> dict[str, Any] | None:
    reflection = _read_json_if_exists(bundle / "artifacts" / "agent_reflection.json")
    if not reflection and not agent_memory:
        return None
    reusable = []
    next_actions = []
    outcome = None
    if reflection:
        reusable = reflection.get("reusable_memory", [])
        next_actions = reflection.get("next_actions", [])
        outcome = reflection.get("outcome")
    elif agent_memory:
        reusable = agent_memory.get("reusable_notes", [])
        outcome = agent_memory.get("outcome")
    text_parts = [item for item in [*(reusable or []), *(next_actions or [])] if isinstance(item, str)]
    return {
        "run_id": run_id,
        "outcome": outcome,
        "reusable_memory": reusable if isinstance(reusable, list) else [],
        "next_actions": next_actions if isinstance(next_actions, list) else [],
        "retrieval_text": " ".join(text_parts),
        "path": "artifacts/agent_reflection.json" if reflection else "memory/agent_memory.json",
    }


def _run_record(bundle: Path, manifest: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    run_id = str(ledger.get("run_id") or manifest.get("run_id") or bundle.name)
    experiment_memory = _experiment_memory_for_run(bundle, run_id)
    agent_memory = _agent_memory_for_run(bundle, run_id)
    fingerprints = _input_fingerprints(bundle, manifest, ledger, experiment_memory)
    evaluation = _evaluation_summary(bundle)
    reflection = _agent_reflection(bundle, run_id, agent_memory)
    metrics = ledger.get("metrics") if isinstance(ledger.get("metrics"), dict) else {}
    if not metrics and evaluation:
        metrics = evaluation.get("metrics", {})
    retrieval_keys = []
    if experiment_memory and isinstance(experiment_memory.get("retrieval_keys"), list):
        retrieval_keys.extend(key for key in experiment_memory["retrieval_keys"] if isinstance(key, str))
    if agent_memory and isinstance(agent_memory.get("experiment_memory"), dict):
        keys = agent_memory["experiment_memory"].get("retrieval_keys", [])
        retrieval_keys.extend(key for key in keys if isinstance(key, str))
    retrieval_keys.extend(
        key
        for key in [
            ledger.get("benchmark_id"),
            ledger.get("dataset_id"),
            ledger.get("model_id"),
            manifest.get("task_id"),
        ]
        if isinstance(key, str)
    )
    return {
        "schema_version": "aira.memory_run_summary.v1",
        "run_id": run_id,
        "task_id": ledger.get("task_id") or manifest.get("task_id"),
        "created_at": ledger.get("created_at") or manifest.get("created_at"),
        "status": str(ledger.get("status", "unknown")),
        "bundle_path": str(bundle),
        "bundle_type": ledger.get("bundle_type") or manifest.get("bundle_type"),
        "benchmark_id": ledger.get("benchmark_id") or manifest.get("benchmark_id"),
        "dataset_id": ledger.get("dataset_id") or manifest.get("dataset_id"),
        "model_id": ledger.get("model_id") or manifest.get("model_id"),
        "metrics": metrics,
        "input_fingerprints": fingerprints,
        "reproducibility": ledger.get("reproducibility", {}),
        "artifacts": ledger.get("artifacts", []),
        "retrieval_keys": sorted(set(retrieval_keys)),
        "experiment_memory": experiment_memory or {},
        "agent_memory": agent_memory or {},
        "agent_reflection": reflection or {},
        "evaluation": evaluation,
    }


def _metric_value(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _build_outcomes(runs: list[dict[str, Any]]) -> dict[str, Any]:
    matrix: dict[tuple[str, str], dict[str, Any]] = {}
    by_dataset: dict[str, dict[str, Any]] = {}
    by_model: dict[str, dict[str, Any]] = {}
    for run in runs:
        dataset_id = str(run.get("dataset_id") or "unknown")
        model_id = str(run.get("model_id") or "unknown")
        status = str(run.get("status"))
        metrics = run.get("metrics") if isinstance(run.get("metrics"), dict) else {}
        accuracy = _metric_value(metrics, "accuracy", "primary_accuracy")
        if accuracy is None and isinstance(run.get("evaluation"), dict):
            evaluation_metrics = run["evaluation"].get("metrics", {})
            if isinstance(evaluation_metrics, dict):
                accuracy = _metric_value(evaluation_metrics, "accuracy", "primary_accuracy")
        key = (dataset_id, model_id)
        entry = matrix.setdefault(
            key,
            {
                "dataset_id": dataset_id,
                "model_id": model_id,
                "run_ids": [],
                "run_count": 0,
                "passed_count": 0,
                "failed_count": 0,
                "best_accuracy": None,
                "latest_status": status,
            },
        )
        entry["run_ids"].append(run["run_id"])
        entry["run_count"] += 1
        entry["passed_count"] += int(status == "passed")
        entry["failed_count"] += int(status == "failed")
        entry["latest_status"] = status
        if accuracy is not None:
            entry["best_accuracy"] = accuracy if entry["best_accuracy"] is None else max(entry["best_accuracy"], accuracy)
        for grouping, grouping_key in [(by_dataset, dataset_id), (by_model, model_id)]:
            summary = grouping.setdefault(
                grouping_key,
                {"run_ids": [], "run_count": 0, "passed_count": 0, "failed_count": 0},
            )
            summary["run_ids"].append(run["run_id"])
            summary["run_count"] += 1
            summary["passed_count"] += int(status == "passed")
            summary["failed_count"] += int(status == "failed")
    return {
        "by_dataset": by_dataset,
        "by_model": by_model,
        "matrix": sorted(matrix.values(), key=lambda item: (item["dataset_id"], item["model_id"])),
    }


def _build_fingerprint_index(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_fingerprint: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    by_run: dict[str, dict[str, str]] = {}
    for run in runs:
        fingerprints = run.get("input_fingerprints") if isinstance(run.get("input_fingerprints"), dict) else {}
        by_run[run["run_id"]] = dict(fingerprints)
        for name, value in fingerprints.items():
            by_fingerprint[name][value].append(run["run_id"])
    return {
        "by_run": by_run,
        "by_fingerprint": {
            name: {value: sorted(run_ids) for value, run_ids in values.items()}
            for name, values in sorted(by_fingerprint.items())
        },
    }


def _build_failures(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for run in runs:
        if run["status"] == "failed":
            failures.append(
                {
                    "schema_version": "aira.failure_ledger_entry.v1",
                    "run_id": run["run_id"],
                    "failure_kind": "run_failed",
                    "status": run["status"],
                    "bundle_path": run["bundle_path"],
                    "task_id": run.get("task_id"),
                    "dataset_id": run.get("dataset_id"),
                    "model_id": run.get("model_id"),
                    "input_fingerprints": run.get("input_fingerprints", {}),
                    "count": 1,
                }
            )
        task_summary = _read_json_if_exists(Path(run["bundle_path"]) / "artifacts" / "task_summary.json")
        if task_summary:
            for task in task_summary.get("tasks", []):
                if not isinstance(task, dict) or task.get("status") not in {"failed", "skipped"}:
                    continue
                failures.append(
                    {
                        "schema_version": "aira.failure_ledger_entry.v1",
                        "run_id": run["run_id"],
                        "failure_kind": "task_" + str(task.get("status")),
                        "task_id": task.get("task_id"),
                        "status": task.get("status"),
                        "bundle_path": run["bundle_path"],
                        "stderr": task.get("stderr"),
                        "missing_outputs": task.get("missing_outputs", []),
                        "count": 1,
                    }
                )
        evaluation = run.get("evaluation") if isinstance(run.get("evaluation"), dict) else {}
        for taxonomy in evaluation.get("taxonomy", []):
            if not isinstance(taxonomy, dict):
                continue
            failures.append(
                {
                    "schema_version": "aira.failure_ledger_entry.v1",
                    "run_id": run["run_id"],
                    "failure_kind": "evaluation_error_taxonomy",
                    "error_type": taxonomy.get("error_type"),
                    "description": taxonomy.get("description"),
                    "count": taxonomy.get("count", 0),
                    "bundle_path": run["bundle_path"],
                    "dataset_id": run.get("dataset_id"),
                    "model_id": run.get("model_id"),
                }
            )
        memory = run.get("experiment_memory") if isinstance(run.get("experiment_memory"), dict) else {}
        analysis = memory.get("error_analysis") if isinstance(memory.get("error_analysis"), dict) else {}
        if analysis.get("ablation_error_count", 0):
            failures.append(
                {
                    "schema_version": "aira.failure_ledger_entry.v1",
                    "run_id": run["run_id"],
                    "failure_kind": "ablation_regression",
                    "error_type": analysis.get("dominant_ablation_error_type"),
                    "count": analysis.get("ablation_error_count"),
                    "bundle_path": run["bundle_path"],
                    "dataset_id": run.get("dataset_id"),
                    "model_id": run.get("model_id"),
                }
            )
    return failures


def _build_retrieval(runs: list[dict[str, Any]]) -> dict[str, Any]:
    keys: dict[str, list[str]] = defaultdict(list)
    reflections: list[dict[str, Any]] = []
    for run in runs:
        for key in run.get("retrieval_keys", []):
            keys[str(key)].append(run["run_id"])
        reflection = run.get("agent_reflection")
        if isinstance(reflection, dict) and reflection:
            reflections.append(reflection)
    return {
        "keys": {key: sorted(set(run_ids)) for key, run_ids in sorted(keys.items())},
        "reflections": sorted(reflections, key=lambda item: item["run_id"]),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_memory_index(
    runs: Iterable[str | Path],
    out: str | Path,
    *,
    status_filter: str = "all",
    max_runs: int | None = None,
    reset: bool = True,
) -> dict[str, Any]:
    """Build a local cross-run index from one or more AIRA result bundle roots."""
    output = Path(out).expanduser().resolve()
    if reset and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    run_records: list[dict[str, Any]] = []
    source_roots = [str(Path(root).expanduser().resolve()) for root in runs]
    for bundle in _bundle_dirs(runs):
        manifest = _read_json(bundle / "bundle_manifest.json")
        for ledger in _ledger_entries(bundle, manifest):
            record = _run_record(bundle, manifest, ledger)
            if status_filter != "all" and record["status"] != status_filter:
                continue
            run_records.append(record)

    run_records.sort(key=lambda item: (str(item.get("created_at")), item["run_id"], item["bundle_path"]))
    if max_runs is not None:
        run_records = run_records[-max_runs:]

    fingerprints = _build_fingerprint_index(run_records)
    outcomes = _build_outcomes(run_records)
    failures = _build_failures(run_records)
    retrieval = _build_retrieval(run_records)
    lifecycle = {
        "mode": "rebuild" if reset else "overwrite_files",
        "status_filter": status_filter,
        "max_runs": max_runs,
        "source_roots": source_roots,
        "indexed_at": _created_at(),
        "retention_policy": "all_matching_runs" if max_runs is None else f"latest_{max_runs}_matching_runs",
    }
    index = {
        "schema_version": MEMORY_INDEX_SCHEMA_VERSION,
        "task_id": MEMORY_TASK_ID,
        "status": "passed",
        "output_dir": str(output),
        "source_roots": source_roots,
        "run_count": len(run_records),
        "failure_count": len(failures),
        "reflection_count": len(retrieval["reflections"]),
        "lifecycle": lifecycle,
        "runs": run_records,
        "failures": failures,
        "fingerprints": fingerprints,
        "outcomes": outcomes,
        "retrieval": retrieval,
        "artifacts": {
            "index": "memory_index.json",
            "runs": "runs.jsonl",
            "failures": "failures.jsonl",
            "fingerprints": "fingerprints.json",
            "outcomes": "outcomes.json",
            "reflections": "reflections.json",
        },
    }
    write_json(output / "memory_index.json", index)
    _write_jsonl(output / "runs.jsonl", run_records)
    _write_jsonl(output / "failures.jsonl", failures)
    write_json(output / "fingerprints.json", fingerprints)
    write_json(output / "outcomes.json", outcomes)
    write_json(output / "reflections.json", retrieval)
    return index
