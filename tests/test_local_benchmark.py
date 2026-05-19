import json

from aira import cli
from aira.benchmark import (
    LOCAL_BENCHMARK_ID,
    LOCAL_DATASET_ID,
    LOCAL_MODEL_ID,
    build_local_provenance,
    evaluate_local_benchmark,
    write_local_benchmark_bundle,
)
from aira.bundles import validate_bundle
from aira.registries import registry_payload


def test_local_benchmark_metrics_are_deterministic():
    payload = evaluate_local_benchmark()

    assert payload["schema_version"] == "aira.local_benchmark.v1"
    assert payload["benchmark_id"] == LOCAL_BENCHMARK_ID
    assert payload["dataset_id"] == LOCAL_DATASET_ID
    assert payload["model_id"] == LOCAL_MODEL_ID
    assert payload["row_count"] == 8
    assert payload["metrics"] == {
        "accuracy": 1.0,
        "macro_f1": 1.0,
        "baseline_accuracy": 0.5,
        "baseline_macro_f1": 0.333333,
        "accuracy_delta": 0.5,
    }
    assert payload["deterministic"] is True
    assert payload["network_required"] is False
    assert payload["external_datasets_required"] is False
    assert payload["gpu_required"] is False
    assert payload["live_model_calls"] is False


def test_local_benchmark_provenance_is_reproducible():
    first = build_local_provenance()
    second = build_local_provenance()

    assert first == second
    assert first["schema_version"] == "aira.benchmark_provenance.v1"
    assert first["run_id"].startswith("aira-local-")
    assert first["determinism"] == {
        "deterministic": True,
        "random_seed": None,
        "network_required": False,
        "external_datasets_required": False,
        "gpu_required": False,
        "live_model_calls": False,
    }
    assert len(first["input_fingerprints"]["dataset_sha256"]) == 64
    assert len(first["input_fingerprints"]["model_config_sha256"]) == 64


def test_write_local_benchmark_bundle_persists_provenance_and_run_ledger(tmp_path):
    output = tmp_path / "aira_local_bundle"

    payload = write_local_benchmark_bundle(output)

    assert payload["status"] == "passed"
    assert payload["validation"]["valid"] is True
    assert payload["validation"]["metadata"]["provenance_artifacts"] == ["artifacts/provenance.json"]
    assert payload["validation"]["metadata"]["run_ledger_artifacts"] == [
        "artifacts/run_ledger_entry.json",
        "memory/run_ledger.jsonl",
    ]
    assert payload["validation"]["metadata"]["run_ledger_entry_count"] == 1
    assert payload["validation"]["metadata"]["run_ledger_run_ids"] == [payload["run_id"]]

    provenance = json.loads((output / "artifacts" / "provenance.json").read_text(encoding="utf-8"))
    ledger_entry = json.loads((output / "artifacts" / "run_ledger_entry.json").read_text(encoding="utf-8"))
    ledger_lines = (output / "memory" / "run_ledger.jsonl").read_text(encoding="utf-8").splitlines()

    assert provenance["run_id"] == payload["run_id"]
    assert ledger_entry["run_id"] == payload["run_id"]
    assert ledger_entry["status"] == "passed"
    assert ledger_entry["reproducibility"]["live_model_calls"] is False
    assert len(ledger_lines) == 1
    assert json.loads(ledger_lines[0]) == ledger_entry


def test_local_benchmark_bundle_validates(tmp_path):
    output = tmp_path / "bundle"
    write_local_benchmark_bundle(output)

    result = validate_bundle(output)

    assert result.valid
    assert result.metadata["artifact_count"] == 9
    assert result.metadata["claim_count"] == 1


def test_local_benchmark_cli_emits_json(tmp_path, capsys):
    output = tmp_path / "cli_bundle"

    exit_code = cli.main(["run-local-benchmark", "--out", str(output), "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "passed"
    assert payload["benchmark"]["metrics"]["accuracy_delta"] == 0.5
    assert payload["run_ledger"]["entry"]["status"] == "passed"


def test_local_benchmark_is_registered():
    payload = registry_payload()

    dataset_ids = {item["id"] for item in payload["datasets"]}
    model_ids = {item["id"] for item in payload["models"]}
    benchmarks = {item["id"]: item for item in payload["benchmarks"]}

    assert LOCAL_DATASET_ID in dataset_ids
    assert LOCAL_MODEL_ID in model_ids
    assert "deterministic-pass-prior-baseline-v1" in model_ids
    assert benchmarks[LOCAL_BENCHMARK_ID]["entrypoint"] == "python3 -m aira run-local-benchmark"
    assert benchmarks[LOCAL_BENCHMARK_ID]["emits_artifact_kinds"] == [
        "benchmark_report",
        "provenance",
        "run_ledger",
    ]
