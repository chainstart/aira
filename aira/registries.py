"""Registries for AIRA datasets, models, and benchmarks."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DATASETS: list[dict[str, Any]] = [
    {
        "id": "fixture-ai-classification",
        "name": "Deterministic fixture classification set",
        "status": "placeholder",
        "source": "local_synthetic",
        "rows": 6,
        "network_required": False,
        "intended_use": "Smoke-test bundle emission and validator contracts.",
    },
    {
        "id": "local-experiment-outcomes-v1",
        "name": "Local experiment outcome text classification set",
        "status": "local_deterministic",
        "source": "builtin_local_fixture",
        "rows": 12,
        "label_set": ["fail", "pass"],
        "splits": ["core", "handoff"],
        "network_required": False,
        "external_datasets_required": False,
        "intended_use": "Exercise local benchmark execution, provenance, ablation analysis, and run memory.",
    },
]

MODELS: list[dict[str, Any]] = [
    {
        "id": "fixture-threshold-classifier",
        "name": "Fixture threshold classifier",
        "status": "placeholder",
        "implementation": "aira.benchmark.threshold_predict",
        "live_model_calls": False,
        "intended_use": "Deterministic benchmark smoke only.",
    },
    {
        "id": "fixture-majority-baseline",
        "name": "Fixture majority baseline",
        "status": "placeholder",
        "implementation": "aira.benchmark.majority_predict",
        "live_model_calls": False,
        "intended_use": "Local comparison baseline for fixture smoke.",
    },
    {
        "id": "deterministic-keyword-outcome-classifier-v1",
        "name": "Deterministic keyword outcome classifier",
        "status": "local_deterministic",
        "implementation": "aira.benchmark.keyword_outcome_predict",
        "live_model_calls": False,
        "network_required": False,
        "gpu_required": False,
        "intended_use": "Local text classification benchmark runner without external dependencies.",
    },
    {
        "id": "deterministic-pass-prior-baseline-v1",
        "name": "Deterministic pass-prior baseline",
        "status": "local_deterministic",
        "implementation": "aira.benchmark.pass_prior_predict",
        "live_model_calls": False,
        "network_required": False,
        "gpu_required": False,
        "intended_use": "Deterministic baseline for the local text outcome benchmark.",
    },
    {
        "id": "deterministic-keyword-no-negative-ablation-v1",
        "name": "Deterministic keyword classifier without negative terms",
        "status": "local_deterministic_ablation",
        "implementation": "aira.benchmark.keyword_no_negative_predict",
        "live_model_calls": False,
        "network_required": False,
        "gpu_required": False,
        "intended_use": "Local ablation fixture proving negative outcome terms are required for fail examples.",
    },
]

BENCHMARKS: list[dict[str, Any]] = [
    {
        "id": "fixture-classification-smoke",
        "name": "AIRA deterministic fixture benchmark",
        "status": "mvp",
        "dataset_id": "fixture-ai-classification",
        "model_ids": ["fixture-threshold-classifier", "fixture-majority-baseline"],
        "metric_ids": ["accuracy", "accuracy_delta"],
        "network_required": False,
        "emits_bundle_type": "aira_result_bundle",
    },
    {
        "id": "local-text-outcome-classification",
        "name": "AIRA deterministic local text outcome benchmark",
        "status": "local_deterministic",
        "dataset_id": "local-experiment-outcomes-v1",
        "model_ids": [
            "deterministic-keyword-outcome-classifier-v1",
            "deterministic-pass-prior-baseline-v1",
            "deterministic-keyword-no-negative-ablation-v1",
        ],
        "metric_ids": ["accuracy", "macro_f1", "baseline_accuracy", "accuracy_delta", "ablation_error_count"],
        "network_required": False,
        "external_datasets_required": False,
        "gpu_required": False,
        "live_model_calls": False,
        "emits_bundle_type": "aira_result_bundle",
        "emits_artifact_kinds": [
            "benchmark_report",
            "ablation_report",
            "error_analysis",
            "provenance",
            "run_ledger",
            "experiment_memory",
        ],
        "entrypoint": "python3 -m aira run-local-benchmark",
    },
]


def registry_payload() -> dict[str, Any]:
    """Return registry placeholders as JSON-serializable data."""
    return {
        "schema_version": "aira.registry.v1",
        "datasets": deepcopy(DATASETS),
        "models": deepcopy(MODELS),
        "benchmarks": deepcopy(BENCHMARKS),
    }
