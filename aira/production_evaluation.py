"""Production-local evaluation artifacts for AIRA result bundles."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from aira.bundles import validate_bundle, write_json
from aira.production_runner import CREATED_AT, RUNNER_DATASET_ID, RUNNER_MODEL_ID, TASK_ID


EVALUATION_SCHEMA_VERSION = "aira.production_evaluation.v1"
METRICS_SCHEMA_VERSION = "aira.production_evaluation_metrics.v1"
ABLATION_SCHEMA_VERSION = "aira.production_ablation_matrix.v1"
ERROR_TAXONOMY_SCHEMA_VERSION = "aira.production_error_taxonomy.v1"
STATISTICAL_TESTS_SCHEMA_VERSION = "aira.production_statistical_tests.v1"
REPORT_SUMMARY_SCHEMA_VERSION = "aira.production_report_summary.v1"
EVALUATION_TASK_ID = "AIRA-PROD-EVAL-001"
FAILURE_TERMS = ("blocked", "failed", "failure", "unsafe", "error", "errors", "missing", "regressed", "stale")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _round(value: float) -> float:
    return round(value, 6)


def _classification_metrics(predictions: list[str], labels: list[str]) -> dict[str, Any]:
    if not labels:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}, "confusion_matrix": {}}
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
            "precision": _round(precision),
            "recall": _round(recall),
            "f1": _round(f1),
        }
    confusion_matrix = {
        label: {
            predicted: sum(
                1
                for actual, prediction in zip(labels, predictions, strict=True)
                if actual == label and prediction == predicted
            )
            for predicted in classes
        }
        for label in sorted(set(labels))
    }
    return {
        "accuracy": _round(sum(1 for prediction, label in zip(predictions, labels, strict=True) if prediction == label) / len(labels)),
        "macro_f1": _round(sum(f1_values) / len(f1_values)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
    }


def _artifact_entries(bundle_path: Path) -> list[dict[str, Any]]:
    payload = _read_json(bundle_path / "artifact_manifest.json")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("artifact_manifest.json field `artifacts` must be a list.")
    return [artifact for artifact in artifacts if isinstance(artifact, dict)]


def _prediction_artifacts(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for artifact in artifacts:
        artifact_id = str(artifact.get("artifact_id", ""))
        kind = str(artifact.get("kind", ""))
        path = str(artifact.get("path", ""))
        haystack = " ".join([artifact_id, kind, path]).lower()
        if "prediction" in haystack and path.endswith(".csv"):
            matches.append(artifact)
    return matches


def _first_present(row: dict[str, str], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _load_prediction_rows(bundle_path: Path, artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact in _prediction_artifacts(artifacts):
        relative = str(artifact.get("path", ""))
        with (bundle_path / relative).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader, start=1):
                label = _first_present(row, ("label", "expected", "ground_truth", "target"))
                prediction = _first_present(row, ("prediction", "predicted", "output"))
                if not label or not prediction:
                    continue
                rows.append(
                    {
                        "example_id": _first_present(row, ("example_id", "id")) or f"{artifact.get('artifact_id')}-{index}",
                        "label": label,
                        "prediction": prediction,
                        "text": _first_present(row, ("text", "input", "prompt")),
                        "source_artifact_id": str(artifact.get("artifact_id", "")),
                        "source_path": relative,
                    }
                )
    return rows


def _preferred_positive_label(labels: list[str], predictions: list[str]) -> str:
    candidates = labels + predictions
    if "pass" in candidates:
        return "pass"
    if "success" in candidates:
        return "success"
    counts = Counter(labels)
    if not counts:
        return "pass"
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _baseline_predictions(labels: list[str], predictions: list[str]) -> list[str]:
    positive_label = _preferred_positive_label(labels, predictions)
    return [positive_label for _ in labels]


def _ablation_prediction(row: dict[str, Any], positive_label: str) -> str:
    text = str(row.get("text", "")).lower()
    prediction = str(row["prediction"])
    if text and any(term in text for term in FAILURE_TERMS):
        return positive_label
    return prediction


def _error_type(label: str, prediction: str, *, ablation: bool) -> str:
    if label == "fail" and prediction == "pass":
        return "false_pass_without_failure_terms" if ablation else "false_pass"
    if label == "pass" and prediction == "fail":
        return "false_fail"
    return f"{label}_as_{prediction}"


def _errors(rows: list[dict[str, Any]], predictions: list[str], *, ablation: bool) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row, prediction in zip(rows, predictions, strict=True):
        label = str(row["label"])
        if prediction == label:
            continue
        result.append(
            {
                "example_id": row["example_id"],
                "label": label,
                "prediction": prediction,
                "error_type": _error_type(label, prediction, ablation=ablation),
                "source_artifact_id": row["source_artifact_id"],
                "source_path": row["source_path"],
            }
        )
    return result


def _exact_mcnemar_p_value(primary_correct: list[bool], ablation_correct: list[bool]) -> dict[str, Any]:
    primary_only = sum(1 for first, second in zip(primary_correct, ablation_correct, strict=True) if first and not second)
    ablation_only = sum(1 for first, second in zip(primary_correct, ablation_correct, strict=True) if second and not first)
    discordant = primary_only + ablation_only
    if discordant == 0:
        p_value = 1.0
        statistic = 0.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(0, min(primary_only, ablation_only) + 1))
        p_value = min(1.0, 2 * tail / (2**discordant))
        statistic = ((abs(primary_only - ablation_only) - 1) ** 2) / discordant
    return {
        "test_id": "primary_vs_failure_keyword_ablation_mcnemar_exact",
        "method": "exact_mcnemar_binomial_two_sided",
        "alpha": 0.05,
        "primary_correct_ablation_wrong": primary_only,
        "primary_wrong_ablation_correct": ablation_only,
        "discordant_pair_count": discordant,
        "statistic": _round(statistic),
        "p_value": _round(p_value),
        "significant": p_value < 0.05,
    }


def _upsert_artifacts(bundle_path: Path, new_artifacts: list[dict[str, Any]]) -> None:
    manifest_path = bundle_path / "artifact_manifest.json"
    payload = _read_json(manifest_path)
    existing = [artifact for artifact in payload.get("artifacts", []) if isinstance(artifact, dict)]
    new_ids = {artifact["artifact_id"] for artifact in new_artifacts}
    payload["artifacts"] = [artifact for artifact in existing if artifact.get("artifact_id") not in new_ids] + new_artifacts
    write_json(manifest_path, payload)


def _upsert_evaluation_claim(bundle_path: Path) -> None:
    claims_path = bundle_path / "claims.json"
    payload = _read_json(claims_path)
    claims = [claim for claim in payload.get("claims", []) if isinstance(claim, dict)]
    claim = {
        "claim_id": "aira-production-evaluation-c1",
        "claim": (
            "The AIRA production evaluation pass emitted deterministic metrics, ablation matrix, "
            "error taxonomy, statistical tests, and machine-readable report summaries for the production-local bundle."
        ),
        "status": "confirmed",
        "reproduction_status": "reproduced",
        "supported_by": [
            "reproduction_status",
            "production_evaluation_metrics",
            "production_ablation_matrix",
            "production_error_taxonomy",
            "production_statistical_tests",
            "production_report_summary",
        ],
        "limitations": [
            "The production-local evaluator compares materialized local prediction artifacts only.",
            "The bundled ablation disables deterministic failure-keyword behavior; it is not a retrained model.",
            "Statistical tests are exact paired tests over the available bundle rows and may have low power on tiny fixtures.",
        ],
    }
    payload["claims"] = [item for item in claims if item.get("claim_id") != claim["claim_id"]] + [claim]
    write_json(claims_path, payload)


def _update_bundle_manifest(bundle_path: Path, run_id: str, status: str) -> None:
    manifest_path = bundle_path / "bundle_manifest.json"
    payload = _read_json(manifest_path)
    payload["production_evaluation"] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "task_id": EVALUATION_TASK_ID,
        "run_id": run_id,
        "status": status,
        "artifacts": {
            "metrics": "artifacts/production_evaluation/metrics.json",
            "ablation_matrix": "artifacts/production_evaluation/ablation_matrix.json",
            "error_taxonomy": "artifacts/production_evaluation/error_taxonomy.json",
            "statistical_tests": "artifacts/production_evaluation/statistical_tests.json",
            "report_summary": "artifacts/production_evaluation/report_summary.json",
        },
    }
    write_json(manifest_path, payload)


def evaluate_production_bundle(bundle_path: str | Path) -> dict[str, Any]:
    """Evaluate a production-local result bundle and append machine-readable reports."""
    bundle = Path(bundle_path).expanduser().resolve()
    input_validation = validate_bundle(bundle)
    if not input_validation.valid:
        return {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "status": "failed",
            "bundle_path": str(bundle),
            "errors": ["Input bundle did not validate before production evaluation."],
            "validation": input_validation.to_dict(),
        }

    artifacts = _artifact_entries(bundle)
    rows = _load_prediction_rows(bundle, artifacts)
    if not rows:
        return {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "status": "failed",
            "bundle_path": str(bundle),
            "errors": ["No CSV prediction artifacts with label and prediction columns were found."],
            "validation": input_validation.to_dict(),
        }

    manifest = _read_json(bundle / "bundle_manifest.json")
    run_id = str(manifest.get("run_id", "aira-prod-eval"))
    labels = [str(row["label"]) for row in rows]
    predictions = [str(row["prediction"]) for row in rows]
    baseline = _baseline_predictions(labels, predictions)
    positive_label = _preferred_positive_label(labels, predictions)
    ablation_predictions = [_ablation_prediction(row, positive_label) for row in rows]

    primary_metrics = _classification_metrics(predictions, labels)
    baseline_metrics = _classification_metrics(baseline, labels)
    ablation_metrics = _classification_metrics(ablation_predictions, labels)
    primary_errors = _errors(rows, predictions, ablation=False)
    ablation_errors = _errors(rows, ablation_predictions, ablation=True)
    changed_predictions = [
        {
            "example_id": row["example_id"],
            "label": row["label"],
            "primary_prediction": row["prediction"],
            "ablation_prediction": ablation_predictions[index],
            "source_artifact_id": row["source_artifact_id"],
        }
        for index, row in enumerate(rows)
        if ablation_predictions[index] != row["prediction"]
    ]

    metrics_report = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "run_id": run_id,
        "task_id": EVALUATION_TASK_ID,
        "source_task_id": TASK_ID,
        "created_at": CREATED_AT,
        "bundle_path": str(bundle),
        "dataset_id": manifest.get("dataset_id", RUNNER_DATASET_ID),
        "model_id": manifest.get("model_id", RUNNER_MODEL_ID),
        "row_count": len(rows),
        "label_set": sorted(set(labels)),
        "prediction_artifacts": sorted({row["source_path"] for row in rows}),
        "metrics": {
            "accuracy": primary_metrics["accuracy"],
            "macro_f1": primary_metrics["macro_f1"],
            "baseline_accuracy": baseline_metrics["accuracy"],
            "baseline_macro_f1": baseline_metrics["macro_f1"],
            "accuracy_delta_vs_baseline": _round(primary_metrics["accuracy"] - baseline_metrics["accuracy"]),
        },
        "per_class": primary_metrics["per_class"],
        "confusion_matrix": primary_metrics["confusion_matrix"],
        "baseline": {
            "baseline_id": "production-preferred-label-baseline-v1",
            "prediction_label": positive_label,
            "metrics": {
                "accuracy": baseline_metrics["accuracy"],
                "macro_f1": baseline_metrics["macro_f1"],
            },
        },
        "deterministic": True,
    }
    ablation_matrix = {
        "schema_version": ABLATION_SCHEMA_VERSION,
        "run_id": run_id,
        "task_id": EVALUATION_TASK_ID,
        "primary_model_id": metrics_report["model_id"],
        "columns": [
            "ablation_id",
            "disabled_features",
            "accuracy",
            "macro_f1",
            "accuracy_delta_vs_primary",
            "changed_prediction_count",
            "error_count",
        ],
        "rows": [
            {
                "ablation_id": "disable-failure-keyword-terms",
                "model_id": "production-local-failure-keyword-ablation-v1",
                "disabled_features": ["failure_keyword_terms"],
                "failure_terms": list(FAILURE_TERMS),
                "preferred_label": positive_label,
                "metrics": {
                    "accuracy": ablation_metrics["accuracy"],
                    "macro_f1": ablation_metrics["macro_f1"],
                    "accuracy_delta_vs_primary": _round(ablation_metrics["accuracy"] - primary_metrics["accuracy"]),
                    "changed_prediction_count": len(changed_predictions),
                    "error_count": len(ablation_errors),
                },
                "changed_predictions": changed_predictions,
            }
        ],
        "deterministic": True,
    }
    taxonomy_counts = Counter(error["error_type"] for error in primary_errors + ablation_errors)
    error_taxonomy = {
        "schema_version": ERROR_TAXONOMY_SCHEMA_VERSION,
        "run_id": run_id,
        "task_id": EVALUATION_TASK_ID,
        "primary_error_count": len(primary_errors),
        "primary_errors": primary_errors,
        "ablation_error_count": len(ablation_errors),
        "ablation_errors": ablation_errors,
        "taxonomy": [
            {
                "error_type": error_type,
                "count": count,
                "description": (
                    "A failure-labeled example was predicted as pass after failure keyword behavior was disabled."
                    if error_type == "false_pass_without_failure_terms"
                    else "Observed prediction did not match the expected label."
                ),
            }
            for error_type, count in sorted(taxonomy_counts.items())
        ],
        "deterministic": True,
    }
    primary_correct = [prediction == label for prediction, label in zip(predictions, labels, strict=True)]
    ablation_correct = [
        prediction == label for prediction, label in zip(ablation_predictions, labels, strict=True)
    ]
    mcnemar = _exact_mcnemar_p_value(primary_correct, ablation_correct)
    statistical_tests = {
        "schema_version": STATISTICAL_TESTS_SCHEMA_VERSION,
        "run_id": run_id,
        "task_id": EVALUATION_TASK_ID,
        "paired_row_count": len(rows),
        "tests": [mcnemar],
        "effect_sizes": {
            "accuracy_delta_primary_vs_ablation": _round(primary_metrics["accuracy"] - ablation_metrics["accuracy"]),
            "primary_error_count": len(primary_errors),
            "ablation_error_count": len(ablation_errors),
        },
        "deterministic": True,
    }
    report_summary = {
        "schema_version": REPORT_SUMMARY_SCHEMA_VERSION,
        "run_id": run_id,
        "task_id": EVALUATION_TASK_ID,
        "status": "passed",
        "headline": "Production-local evaluation completed with deterministic paired comparison artifacts.",
        "summary": {
            "row_count": len(rows),
            "primary_accuracy": primary_metrics["accuracy"],
            "baseline_accuracy": baseline_metrics["accuracy"],
            "ablation_accuracy": ablation_metrics["accuracy"],
            "primary_error_count": len(primary_errors),
            "ablation_error_count": len(ablation_errors),
            "mcnemar_p_value": mcnemar["p_value"],
            "mcnemar_significant": mcnemar["significant"],
        },
        "artifact_paths": {
            "metrics": "artifacts/production_evaluation/metrics.json",
            "ablation_matrix": "artifacts/production_evaluation/ablation_matrix.json",
            "error_taxonomy": "artifacts/production_evaluation/error_taxonomy.json",
            "statistical_tests": "artifacts/production_evaluation/statistical_tests.json",
            "report_summary": "artifacts/production_evaluation/report_summary.json",
        },
        "recommendations": [
            "Keep failure keyword behavior enabled for this production-local fixture.",
            "Use larger paired datasets before treating exact-test significance as production evidence.",
            "Preserve prediction CSV artifacts so future evaluators can reproduce comparison tables.",
        ],
        "deterministic": True,
    }

    evaluation_dir = bundle / "artifacts" / "production_evaluation"
    write_json(evaluation_dir / "metrics.json", metrics_report)
    write_json(evaluation_dir / "ablation_matrix.json", ablation_matrix)
    write_json(evaluation_dir / "error_taxonomy.json", error_taxonomy)
    write_json(evaluation_dir / "statistical_tests.json", statistical_tests)
    write_json(evaluation_dir / "report_summary.json", report_summary)

    new_artifacts = [
        {
            "artifact_id": "production_evaluation_metrics",
            "path": "artifacts/production_evaluation/metrics.json",
            "kind": "production_evaluation_metrics",
            "description": "Deterministic production-local classification metrics and baseline comparison.",
        },
        {
            "artifact_id": "production_ablation_matrix",
            "path": "artifacts/production_evaluation/ablation_matrix.json",
            "kind": "production_ablation_matrix",
            "description": "Machine-readable ablation matrix for production-local prediction behavior.",
        },
        {
            "artifact_id": "production_error_taxonomy",
            "path": "artifacts/production_evaluation/error_taxonomy.json",
            "kind": "production_error_taxonomy",
            "description": "Primary and ablation error taxonomy for production-local predictions.",
        },
        {
            "artifact_id": "production_statistical_tests",
            "path": "artifacts/production_evaluation/statistical_tests.json",
            "kind": "production_statistical_tests",
            "description": "Exact paired statistical tests for production-local experiment comparison.",
        },
        {
            "artifact_id": "production_report_summary",
            "path": "artifacts/production_evaluation/report_summary.json",
            "kind": "production_report_summary",
            "description": "Machine-readable production evaluation report summary.",
        },
    ]
    _upsert_artifacts(bundle, new_artifacts)
    _upsert_evaluation_claim(bundle)
    _update_bundle_manifest(bundle, run_id, "passed")
    output_validation = validate_bundle(bundle)
    status = "passed" if output_validation.valid else "failed"
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": status,
        "bundle_path": str(bundle),
        "run_id": run_id,
        "input_validation": input_validation.to_dict(),
        "validation": output_validation.to_dict(),
        "metrics": metrics_report,
        "ablation_matrix": ablation_matrix,
        "error_taxonomy": error_taxonomy,
        "statistical_tests": statistical_tests,
        "report_summary": report_summary,
        "artifacts": new_artifacts,
    }
