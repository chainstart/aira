"""Deterministic fixture benchmark for the AIRA bootstrap."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from aira.bundles import BUNDLE_SCHEMA_VERSION, validate_bundle, write_json
from aira.registries import registry_payload


CREATED_AT = "2026-05-18T00:00:00Z"
BENCHMARK_ID = "fixture-classification-smoke"
DATASET_ID = "fixture-ai-classification"
MODEL_ID = "fixture-threshold-classifier"
BASELINE_ID = "fixture-majority-baseline"

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


def _accuracy(predictions: list[int], labels: list[int]) -> float:
    correct = sum(1 for prediction, label in zip(predictions, labels, strict=True) if prediction == label)
    return correct / len(labels)


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
