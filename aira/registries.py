"""Placeholder registries for AIRA datasets, models, and benchmarks."""

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
    }
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
    }
]


def registry_payload() -> dict[str, Any]:
    """Return registry placeholders as JSON-serializable data."""
    return {
        "schema_version": "aira.registry.v1",
        "datasets": deepcopy(DATASETS),
        "models": deepcopy(MODELS),
        "benchmarks": deepcopy(BENCHMARKS),
    }
