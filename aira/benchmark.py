"""Deterministic benchmark runners for the AIRA bootstrap."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import sys
from pathlib import Path
from typing import Any

from aira import __version__
from aira.bundles import BUNDLE_SCHEMA_VERSION, validate_bundle, write_json
from aira.registries import registry_payload


CREATED_AT = "2026-05-18T00:00:00Z"
BENCHMARK_ID = "fixture-classification-smoke"
DATASET_ID = "fixture-ai-classification"
MODEL_ID = "fixture-threshold-classifier"
BASELINE_ID = "fixture-majority-baseline"

LOCAL_CREATED_AT = "2026-05-19T00:00:00Z"
LOCAL_BENCHMARK_ID = "local-text-outcome-classification"
LOCAL_DATASET_ID = "local-experiment-outcomes-v1"
LOCAL_MODEL_ID = "deterministic-keyword-outcome-classifier-v1"
LOCAL_BASELINE_ID = "deterministic-pass-prior-baseline-v1"
LOCAL_TASK_ID = "AIRA-BENCHMARK-001"
LOCAL_COMMAND = "python3 -m aira run-local-benchmark"
ARA_HANDOFF_SCHEMA_VERSION = "aira.ara_handoff.v1"
ARA_GATE_PROFILE = "ara-public-bundle-reproduction-gate.v1"
ARA_VALIDATE_COMMAND = "python3 -m aira bundles validate <bundle> --json"

LOCAL_DATASET_ROWS: list[dict[str, str]] = [
    {
        "example_id": "local-001",
        "text": "clear validation passed for the model bundle",
        "label": "pass",
    },
    {
        "example_id": "local-002",
        "text": "benchmark metrics improved after the deterministic fix",
        "label": "pass",
    },
    {
        "example_id": "local-003",
        "text": "reproduction status confirmed without network access",
        "label": "pass",
    },
    {
        "example_id": "local-004",
        "text": "registry snapshot matched the expected schema",
        "label": "pass",
    },
    {
        "example_id": "local-005",
        "text": "missing provenance blocked the experiment handoff",
        "label": "fail",
    },
    {
        "example_id": "local-006",
        "text": "model output regressed on the held out case",
        "label": "fail",
    },
    {
        "example_id": "local-007",
        "text": "flaky setup failed before metrics were written",
        "label": "fail",
    },
    {
        "example_id": "local-008",
        "text": "stale dataset metadata caused validation errors",
        "label": "fail",
    },
]

LOCAL_MODEL_CONFIG: dict[str, Any] = {
    "positive_terms": [
        "clear",
        "confirmed",
        "deterministic",
        "expected",
        "improved",
        "matched",
        "passed",
        "reproduction",
        "validation",
    ],
    "negative_terms": [
        "blocked",
        "caused",
        "errors",
        "failed",
        "flaky",
        "missing",
        "regressed",
        "stale",
    ],
    "tie_break_label": "pass",
}

FIXTURE_ROWS: list[dict[str, Any]] = [
    {"example_id": "fx-001", "score": 0.91, "label": 1},
    {"example_id": "fx-002", "score": 0.83, "label": 1},
    {"example_id": "fx-003", "score": 0.71, "label": 1},
    {"example_id": "fx-004", "score": 0.42, "label": 0},
    {"example_id": "fx-005", "score": 0.24, "label": 0},
    {"example_id": "fx-006", "score": 0.56, "label": 0},
]


def threshold_predict(score: float) -> int:
    return int(score >= 0.5)


def majority_predict(_: float) -> int:
    return 1


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _accuracy(predictions: list[int], labels: list[int]) -> float:
    correct = sum(1 for prediction, label in zip(predictions, labels, strict=True) if prediction == label)
    return correct / len(labels)


def _label_accuracy(predictions: list[str], labels: list[str]) -> float:
    correct = sum(1 for prediction, label in zip(predictions, labels, strict=True) if prediction == label)
    return correct / len(labels)


def _classification_metrics(predictions: list[str], labels: list[str]) -> dict[str, Any]:
    classes = sorted(set(labels) | set(predictions))
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in classes:
        true_positive = sum(
            1
            for prediction, expected in zip(predictions, labels, strict=True)
            if prediction == label and expected == label
        )
        false_positive = sum(
            1
            for prediction, expected in zip(predictions, labels, strict=True)
            if prediction == label and expected != label
        )
        false_negative = sum(
            1
            for prediction, expected in zip(predictions, labels, strict=True)
            if prediction != label and expected == label
        )
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[label] = {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
    macro_f1 = sum(f1_values) / len(f1_values)
    return {
        "accuracy": round(_label_accuracy(predictions, labels), 6),
        "macro_f1": round(macro_f1, 6),
        "per_class": per_class,
    }


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def keyword_outcome_predict(text: str) -> str:
    """Predict pass/fail outcome using a deterministic lexical model."""
    tokens = set(_tokenize(text))
    positive_hits = sum(1 for term in LOCAL_MODEL_CONFIG["positive_terms"] if term in tokens)
    negative_hits = sum(1 for term in LOCAL_MODEL_CONFIG["negative_terms"] if term in tokens)
    if positive_hits > negative_hits:
        return "pass"
    if negative_hits > positive_hits:
        return "fail"
    return str(LOCAL_MODEL_CONFIG["tie_break_label"])


def pass_prior_predict(_: str) -> str:
    return "pass"


def _keyword_score(text: str) -> dict[str, int]:
    tokens = set(_tokenize(text))
    positive_hits = sum(1 for term in LOCAL_MODEL_CONFIG["positive_terms"] if term in tokens)
    negative_hits = sum(1 for term in LOCAL_MODEL_CONFIG["negative_terms"] if term in tokens)
    return {
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
        "margin": positive_hits - negative_hits,
    }


def _local_run_id() -> str:
    digest = _canonical_digest(
        {
            "benchmark_id": LOCAL_BENCHMARK_ID,
            "dataset": LOCAL_DATASET_ROWS,
            "model_config": LOCAL_MODEL_CONFIG,
            "baseline_id": LOCAL_BASELINE_ID,
            "task_id": LOCAL_TASK_ID,
        }
    )
    return f"aira-local-{digest[:12]}"


def build_local_provenance() -> dict[str, Any]:
    registry = registry_payload()
    return {
        "schema_version": "aira.benchmark_provenance.v1",
        "run_id": _local_run_id(),
        "task_id": LOCAL_TASK_ID,
        "created_at": LOCAL_CREATED_AT,
        "benchmark_id": LOCAL_BENCHMARK_ID,
        "dataset_id": LOCAL_DATASET_ID,
        "model_id": LOCAL_MODEL_ID,
        "baseline_model_id": LOCAL_BASELINE_ID,
        "input_fingerprints": {
            "dataset_sha256": _canonical_digest(LOCAL_DATASET_ROWS),
            "model_config_sha256": _canonical_digest(LOCAL_MODEL_CONFIG),
            "registry_snapshot_sha256": _canonical_digest(registry),
        },
        "execution": {
            "runner": "aira.benchmark.write_local_benchmark_bundle",
            "command": LOCAL_COMMAND,
            "package_version": __version__,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "determinism": {
            "deterministic": True,
            "random_seed": None,
            "network_required": False,
            "external_datasets_required": False,
            "gpu_required": False,
            "live_model_calls": False,
        },
        "limitations": [
            "The dataset is a tiny local text fixture for runner and bundle integration.",
            "The lexical classifier is deterministic and does not represent a trained frontier model.",
            "The run ledger is bundle-local; shared experiment memory is reserved for the agent MVP.",
        ],
    }


def build_ara_handoff(report: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    """Describe the deterministic bundle inputs ARA reproduction gates can consume."""
    return {
        "schema_version": ARA_HANDOFF_SCHEMA_VERSION,
        "consumer": "ara",
        "gate_profile": ARA_GATE_PROFILE,
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "bundle_type": "aira_result_bundle",
        "producer": "aira",
        "task_id": LOCAL_TASK_ID,
        "run_id": report["run_id"],
        "created_at": LOCAL_CREATED_AT,
        "status": "ready",
        "validation_command": ARA_VALIDATE_COMMAND,
        "reproduce_command": f"{LOCAL_COMMAND} --out <bundle>",
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
            "benchmark_report": "artifacts/benchmark_report.json",
        },
        "reproducibility": {
            "deterministic": True,
            "network_required": False,
            "external_datasets_required": False,
            "gpu_required": False,
            "live_model_calls": False,
            "input_fingerprints": provenance["input_fingerprints"],
        },
        "claim_gate": {
            "confirmed_claims_require_reproduced_status": True,
            "confirmed_claims_require_reproduction_artifact": True,
        },
        "dispatch": {
            "lab_id": "aira",
            "entrypoint": LOCAL_COMMAND,
            "side_effect_free_validation": True,
            "network_policy": "none",
        },
    }


def evaluate_local_benchmark() -> dict[str, Any]:
    labels = [row["label"] for row in LOCAL_DATASET_ROWS]
    predictions = [keyword_outcome_predict(row["text"]) for row in LOCAL_DATASET_ROWS]
    baseline_predictions = [pass_prior_predict(row["text"]) for row in LOCAL_DATASET_ROWS]
    model_metrics = _classification_metrics(predictions, labels)
    baseline_metrics = _classification_metrics(baseline_predictions, labels)
    confusion_matrix = {
        label: {
            predicted: sum(
                1
                for actual, prediction in zip(labels, predictions, strict=True)
                if actual == label and prediction == predicted
            )
            for predicted in sorted(set(labels) | set(predictions))
        }
        for label in sorted(set(labels))
    }
    return {
        "schema_version": "aira.local_benchmark.v1",
        "run_id": _local_run_id(),
        "benchmark_id": LOCAL_BENCHMARK_ID,
        "dataset_id": LOCAL_DATASET_ID,
        "model_id": LOCAL_MODEL_ID,
        "baseline_model_id": LOCAL_BASELINE_ID,
        "row_count": len(LOCAL_DATASET_ROWS),
        "label_set": sorted(set(labels)),
        "metrics": {
            "accuracy": model_metrics["accuracy"],
            "macro_f1": model_metrics["macro_f1"],
            "baseline_accuracy": baseline_metrics["accuracy"],
            "baseline_macro_f1": baseline_metrics["macro_f1"],
            "accuracy_delta": round(model_metrics["accuracy"] - baseline_metrics["accuracy"], 6),
        },
        "per_class": model_metrics["per_class"],
        "baseline_per_class": baseline_metrics["per_class"],
        "confusion_matrix": confusion_matrix,
        "examples": [
            {
                **row,
                **_keyword_score(row["text"]),
                "prediction": predictions[index],
                "baseline_prediction": baseline_predictions[index],
            }
            for index, row in enumerate(LOCAL_DATASET_ROWS)
        ],
        "deterministic": True,
        "network_required": False,
        "external_datasets_required": False,
        "gpu_required": False,
        "live_model_calls": False,
    }


def evaluate_fixture_benchmark() -> dict[str, Any]:
    labels = [int(row["label"]) for row in FIXTURE_ROWS]
    threshold_predictions = [threshold_predict(float(row["score"])) for row in FIXTURE_ROWS]
    baseline_predictions = [majority_predict(float(row["score"])) for row in FIXTURE_ROWS]
    accuracy = _accuracy(threshold_predictions, labels)
    baseline_accuracy = _accuracy(baseline_predictions, labels)
    return {
        "schema_version": "aira.fixture_benchmark.v1",
        "benchmark_id": BENCHMARK_ID,
        "dataset_id": DATASET_ID,
        "model_id": MODEL_ID,
        "baseline_model_id": BASELINE_ID,
        "row_count": len(FIXTURE_ROWS),
        "metrics": {
            "accuracy": round(accuracy, 6),
            "baseline_accuracy": round(baseline_accuracy, 6),
            "accuracy_delta": round(accuracy - baseline_accuracy, 6),
        },
        "examples": [
            {
                **row,
                "prediction": threshold_predictions[index],
                "baseline_prediction": baseline_predictions[index],
            }
            for index, row in enumerate(FIXTURE_ROWS)
        ],
        "deterministic": True,
        "live_model_calls": False,
    }


def _write_metrics_table(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key in sorted(metrics):
            writer.writerow([key, f"{metrics[key]:.6f}"])


def _write_dataset_table(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["example_id", "score", "label"])
        writer.writeheader()
        writer.writerows(FIXTURE_ROWS)


def _write_local_dataset_table(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["example_id", "text", "label"])
        writer.writeheader()
        writer.writerows(LOCAL_DATASET_ROWS)


def _write_local_predictions_table(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "example_id",
                "label",
                "prediction",
                "baseline_prediction",
                "positive_hits",
                "negative_hits",
                "margin",
                "text",
            ],
        )
        writer.writeheader()
        writer.writerows(examples)


def _build_local_run_ledger_entry(
    *,
    bundle_path: Path,
    report: dict[str, Any],
    provenance: dict[str, Any],
    validation_valid: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "aira.run_ledger_entry.v1",
        "run_id": report["run_id"],
        "task_id": LOCAL_TASK_ID,
        "created_at": LOCAL_CREATED_AT,
        "status": "passed" if validation_valid else "failed",
        "bundle_path": str(bundle_path),
        "bundle_type": "aira_result_bundle",
        "benchmark_id": report["benchmark_id"],
        "dataset_id": report["dataset_id"],
        "model_id": report["model_id"],
        "baseline_model_id": report["baseline_model_id"],
        "metrics": report["metrics"],
        "provenance": {
            "path": "artifacts/provenance.json",
            "dataset_sha256": provenance["input_fingerprints"]["dataset_sha256"],
            "model_config_sha256": provenance["input_fingerprints"]["model_config_sha256"],
        },
        "reproducibility": {
            "deterministic": True,
            "network_required": False,
            "external_datasets_required": False,
            "gpu_required": False,
            "live_model_calls": False,
            "command": LOCAL_COMMAND,
        },
        "artifacts": [
            "artifacts/benchmark_report.json",
            "artifacts/provenance.json",
            "artifacts/reproduction_status.json",
            "memory/run_ledger.jsonl",
        ],
    }


def _write_local_reproducibility_notes(path: Path, report: dict[str, Any], provenance: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Reproducibility Notes",
                "",
                f"- Run id: `{report['run_id']}`.",
                f"- Benchmark: `{LOCAL_BENCHMARK_ID}`.",
                f"- Reproduce locally with `{LOCAL_COMMAND} --out <bundle>`.",
                f"- Validate the emitted bundle with `{ARA_VALIDATE_COMMAND}`.",
                f"- Dataset sha256: `{provenance['input_fingerprints']['dataset_sha256']}`.",
                f"- Model config sha256: `{provenance['input_fingerprints']['model_config_sha256']}`.",
                f"- Registry snapshot sha256: `{provenance['input_fingerprints']['registry_snapshot_sha256']}`.",
                "- The benchmark is deterministic and uses only built-in local fixture data.",
                "- No network dataset, live model API, GPU, training job, or external service is required.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_fixture_bundle(output_dir: str | Path) -> dict[str, Any]:
    out = Path(output_dir).expanduser().resolve()
    artifacts_dir = out / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    report = evaluate_fixture_benchmark()

    write_json(
        out / "bundle_manifest.json",
        {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "bundle_type": "aira_result_bundle",
            "domain": "ai_ml",
            "created_at": CREATED_AT,
            "producer": "aira",
            "benchmark_id": BENCHMARK_ID,
            "deterministic": True,
            "live_model_calls": False,
        },
    )
    write_json(out / "artifacts" / "benchmark_report.json", report)
    write_json(
        out / "artifacts" / "reproduction_status.json",
        {
            "schema_version": "aira.reproduction_status.v1",
            "status": "reproduced",
            "benchmark_id": BENCHMARK_ID,
            "deterministic": True,
            "live_model_calls": False,
            "command": "python3 -m aira run-fixture-benchmark",
            "metrics": report["metrics"],
        },
    )
    write_json(out / "artifacts" / "registry_snapshot.json", registry_payload())
    _write_metrics_table(out / "artifacts" / "metrics_table.csv", report["metrics"])
    _write_dataset_table(out / "artifacts" / "dataset_fixture.csv")
    write_json(
        out / "artifact_manifest.json",
        {
            "artifacts": [
                {
                    "artifact_id": "benchmark_report",
                    "path": "artifacts/benchmark_report.json",
                    "kind": "benchmark_report",
                    "description": "Deterministic fixture benchmark report.",
                },
                {
                    "artifact_id": "reproduction_status",
                    "path": "artifacts/reproduction_status.json",
                    "kind": "reproduction_status",
                    "description": "Local reproduction status for the fixture benchmark.",
                },
                {
                    "artifact_id": "metrics_table",
                    "path": "artifacts/metrics_table.csv",
                    "kind": "metrics_table",
                    "description": "Metrics supporting the fixture benchmark claim.",
                },
                {
                    "artifact_id": "dataset_fixture",
                    "path": "artifacts/dataset_fixture.csv",
                    "kind": "dataset",
                    "description": "Local synthetic fixture data used by the smoke benchmark.",
                },
                {
                    "artifact_id": "registry_snapshot",
                    "path": "artifacts/registry_snapshot.json",
                    "kind": "registry_snapshot",
                    "description": "Dataset, model, and benchmark registry placeholders.",
                },
            ]
        },
    )
    write_json(
        out / "claims.json",
        {
            "claims": [
                {
                    "claim_id": "aira-fixture-c1",
                    "claim": (
                        "The deterministic AIRA fixture threshold model achieved accuracy "
                        "0.833333 and reproduced the expected local benchmark result."
                    ),
                    "status": "confirmed",
                    "reproduction_status": "reproduced",
                    "supported_by": ["reproduction_status", "metrics_table", "benchmark_report"],
                    "limitations": [
                        "The dataset is synthetic and only validates the AIRA experiment bundle contract.",
                        "No live model, GPU training, or external dataset download is performed.",
                    ],
                }
            ]
        },
    )
    (out / "writing_brief.md").write_text(
        "\n".join(
            [
                "# AIRA Fixture Benchmark",
                "",
                "The fixture benchmark is suitable for smoke-testing ARA ingestion of an AI/ML result bundle.",
                "It should not be cited as empirical evidence about real model quality.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (out / "limitations.md").write_text(
        "\n".join(
            [
                "# Limitations",
                "",
                "- Synthetic six-row dataset.",
                "- Deterministic threshold and majority baseline only.",
                "- No live model calls, training, or remote data access.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    validation = validate_bundle(out)
    return {
        "schema_version": "aira.fixture_benchmark_run.v1",
        "status": "passed" if validation.valid else "failed",
        "bundle_path": str(out),
        "benchmark": report,
        "validation": validation.to_dict(),
    }


def write_local_benchmark_bundle(output_dir: str | Path) -> dict[str, Any]:
    out = Path(output_dir).expanduser().resolve()
    artifacts_dir = out / "artifacts"
    memory_dir = out / "memory"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)

    report = evaluate_local_benchmark()
    provenance = build_local_provenance()
    ara_handoff = build_ara_handoff(report, provenance)
    write_json(
        out / "bundle_manifest.json",
        {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "bundle_type": "aira_result_bundle",
            "domain": "ai_ml",
            "created_at": LOCAL_CREATED_AT,
            "producer": "aira",
            "task_id": LOCAL_TASK_ID,
            "run_id": report["run_id"],
            "benchmark_id": LOCAL_BENCHMARK_ID,
            "dataset_id": LOCAL_DATASET_ID,
            "model_id": LOCAL_MODEL_ID,
            "deterministic": True,
            "network_required": False,
            "external_datasets_required": False,
            "gpu_required": False,
            "live_model_calls": False,
            "ara_handoff": {
                "schema_version": ARA_HANDOFF_SCHEMA_VERSION,
                "gate_profile": ARA_GATE_PROFILE,
                "artifact": "artifacts/ara_handoff.json",
                "validation_command": ARA_VALIDATE_COMMAND,
            },
        },
    )
    write_json(out / "artifacts" / "benchmark_report.json", report)
    write_json(out / "artifacts" / "provenance.json", provenance)
    write_json(out / "artifacts" / "ara_handoff.json", ara_handoff)
    _write_local_reproducibility_notes(out / "artifacts" / "reproducibility_notes.md", report, provenance)
    write_json(
        out / "artifacts" / "reproduction_status.json",
        {
            "schema_version": "aira.reproduction_status.v1",
            "status": "reproduced",
            "run_id": report["run_id"],
            "task_id": LOCAL_TASK_ID,
            "benchmark_id": LOCAL_BENCHMARK_ID,
            "dataset_id": LOCAL_DATASET_ID,
            "model_id": LOCAL_MODEL_ID,
            "deterministic": True,
            "network_required": False,
            "external_datasets_required": False,
            "gpu_required": False,
            "live_model_calls": False,
            "command": LOCAL_COMMAND,
            "metrics": report["metrics"],
            "provenance_path": "artifacts/provenance.json",
        },
    )
    write_json(out / "artifacts" / "registry_snapshot.json", registry_payload())
    _write_metrics_table(out / "artifacts" / "metrics_table.csv", report["metrics"])
    _write_local_dataset_table(out / "artifacts" / "dataset_local.csv")
    _write_local_predictions_table(out / "artifacts" / "predictions_table.csv", report["examples"])
    write_json(
        out / "artifact_manifest.json",
        {
            "artifacts": [
                {
                    "artifact_id": "benchmark_report",
                    "path": "artifacts/benchmark_report.json",
                    "kind": "benchmark_report",
                    "description": "Deterministic local text outcome benchmark report.",
                },
                {
                    "artifact_id": "provenance",
                    "path": "artifacts/provenance.json",
                    "kind": "provenance",
                    "description": "Input, execution, and reproducibility provenance for the local benchmark.",
                },
                {
                    "artifact_id": "ara_handoff",
                    "path": "artifacts/ara_handoff.json",
                    "kind": "ara_handoff",
                    "description": "ARA-facing bundle and reproduction gate handoff metadata.",
                },
                {
                    "artifact_id": "reproducibility_notes",
                    "path": "artifacts/reproducibility_notes.md",
                    "kind": "reproducibility_notes",
                    "description": "Human-readable deterministic reproduction notes for public ARA gates.",
                },
                {
                    "artifact_id": "reproduction_status",
                    "path": "artifacts/reproduction_status.json",
                    "kind": "reproduction_status",
                    "description": "Local reproduction status for the deterministic benchmark.",
                },
                {
                    "artifact_id": "metrics_table",
                    "path": "artifacts/metrics_table.csv",
                    "kind": "metrics_table",
                    "description": "Metrics supporting the local benchmark claim.",
                },
                {
                    "artifact_id": "dataset_local",
                    "path": "artifacts/dataset_local.csv",
                    "kind": "dataset",
                    "description": "Tiny local text classification dataset used by the benchmark.",
                },
                {
                    "artifact_id": "predictions_table",
                    "path": "artifacts/predictions_table.csv",
                    "kind": "predictions",
                    "description": "Per-example predictions and lexical hit counts.",
                },
                {
                    "artifact_id": "registry_snapshot",
                    "path": "artifacts/registry_snapshot.json",
                    "kind": "registry_snapshot",
                    "description": "Dataset, model, and benchmark registry snapshot.",
                },
                {
                    "artifact_id": "run_ledger_entry",
                    "path": "artifacts/run_ledger_entry.json",
                    "kind": "run_ledger_entry",
                    "description": "Machine-readable ledger row for experiment agents.",
                },
                {
                    "artifact_id": "run_ledger",
                    "path": "memory/run_ledger.jsonl",
                    "kind": "run_ledger",
                    "description": "Bundle-local appendable run ledger for later experiment agents.",
                },
            ]
        },
    )
    write_json(
        out / "claims.json",
        {
            "claims": [
                {
                    "claim_id": "aira-local-benchmark-c1",
                    "claim": (
                        "The deterministic local AIRA keyword classifier achieved accuracy "
                        "1.000000 on the bundled local text outcome dataset and outperformed "
                        "the pass-prior baseline by 0.500000 accuracy."
                    ),
                    "status": "confirmed",
                    "reproduction_status": "reproduced",
                    "supported_by": [
                        "reproduction_status",
                        "metrics_table",
                        "benchmark_report",
                        "provenance",
                        "ara_handoff",
                        "reproducibility_notes",
                        "run_ledger_entry",
                    ],
                    "limitations": provenance["limitations"],
                }
            ]
        },
    )
    (out / "writing_brief.md").write_text(
        "\n".join(
            [
                "# AIRA Local Benchmark",
                "",
                "This bundle records a deterministic local text outcome classification benchmark.",
                "It is intended to exercise real runner, registry, provenance, and ledger paths without network access.",
                "The lexical classifier and tiny dataset are not evidence of frontier model quality.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (out / "limitations.md").write_text(
        "\n".join(
            [
                "# Limitations",
                "",
                "- Tiny built-in local dataset with eight examples.",
                "- Deterministic lexical classifier and pass-prior baseline only.",
                "- No training, GPU execution, live model API, external dataset, or network access.",
                "- Ledger persistence is bundle-local until the AIRA experiment agent memory service exists.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    ledger_entry = _build_local_run_ledger_entry(
        bundle_path=out,
        report=report,
        provenance=provenance,
        validation_valid=True,
    )
    write_json(out / "artifacts" / "run_ledger_entry.json", ledger_entry)
    (out / "memory" / "run_ledger.jsonl").write_text(
        json.dumps(ledger_entry, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation = validate_bundle(out)
    if not validation.valid:
        ledger_entry = _build_local_run_ledger_entry(
            bundle_path=out,
            report=report,
            provenance=provenance,
            validation_valid=False,
        )
        write_json(out / "artifacts" / "run_ledger_entry.json", ledger_entry)
        (out / "memory" / "run_ledger.jsonl").write_text(
            json.dumps(ledger_entry, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validation = validate_bundle(out)
    return {
        "schema_version": "aira.local_benchmark_run.v1",
        "status": "passed" if validation.valid else "failed",
        "bundle_path": str(out),
        "run_id": report["run_id"],
        "benchmark": report,
        "provenance": {
            "path": str(out / "artifacts" / "provenance.json"),
            "dataset_sha256": provenance["input_fingerprints"]["dataset_sha256"],
            "model_config_sha256": provenance["input_fingerprints"]["model_config_sha256"],
        },
        "run_ledger": {
            "entry_path": str(out / "artifacts" / "run_ledger_entry.json"),
            "ledger_path": str(out / "memory" / "run_ledger.jsonl"),
            "entry": ledger_entry,
        },
        "validation": validation.to_dict(),
    }
