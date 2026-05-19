import json

from aira import cli
from aira.agent import build_agent_plan, run_agent_smoke, select_local_experiment
from aira.benchmark import (
    LOCAL_ABLATION_MODEL_ID,
    LOCAL_BENCHMARK_ID,
    LOCAL_DATASET_ID,
    LOCAL_MODEL_ID,
    build_local_error_analysis,
    build_local_experiment_memory,
    build_local_provenance,
    evaluate_local_ablations,
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
    assert payload["row_count"] == 12
    assert {example["split"] for example in payload["examples"]} == {"core", "handoff"}
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


def test_local_benchmark_ablation_and_error_analysis_are_deterministic():
    report = evaluate_local_benchmark()
    provenance = build_local_provenance()
    ablation = evaluate_local_ablations(report)
    error_analysis = build_local_error_analysis(report, ablation)
    memory = build_local_experiment_memory(report, provenance, ablation, error_analysis)

    assert ablation["schema_version"] == "aira.local_benchmark_ablation.v1"
    assert ablation["ablations"][0]["model_id"] == LOCAL_ABLATION_MODEL_ID
    assert ablation["ablations"][0]["metrics"] == {
        "accuracy": 0.5,
        "macro_f1": 0.333333,
        "accuracy_delta_vs_primary": -0.5,
        "error_count": 6,
    }
    assert error_analysis["schema_version"] == "aira.local_benchmark_error_analysis.v1"
    assert error_analysis["primary_error_count"] == 0
    assert error_analysis["ablation_error_count"] == 6
    assert "negative-term-ablation" in memory["retrieval_keys"]
    assert memory["ablation_findings"][0]["error_count"] == 6


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
    assert payload["validation"]["metadata"]["ara_gate"]["profile"] == "ara-public-bundle-reproduction-gate.v1"
    assert payload["validation"]["metadata"]["ara_gate"]["handoff_artifacts"] == ["artifacts/ara_handoff.json"]
    assert payload["validation"]["metadata"]["ara_gate"]["required_inputs_present"] is True

    provenance = json.loads((output / "artifacts" / "provenance.json").read_text(encoding="utf-8"))
    ara_handoff = json.loads((output / "artifacts" / "ara_handoff.json").read_text(encoding="utf-8"))
    ledger_entry = json.loads((output / "artifacts" / "run_ledger_entry.json").read_text(encoding="utf-8"))
    ledger_lines = (output / "memory" / "run_ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ablation_report = json.loads((output / "artifacts" / "ablation_report.json").read_text(encoding="utf-8"))
    error_analysis = json.loads((output / "artifacts" / "error_analysis.json").read_text(encoding="utf-8"))
    experiment_memory = json.loads((output / "memory" / "experiment_memory.json").read_text(encoding="utf-8"))
    experiment_memory_lines = (output / "memory" / "experiment_memory.jsonl").read_text(encoding="utf-8").splitlines()

    assert provenance["run_id"] == payload["run_id"]
    assert ara_handoff["consumer"] == "ara"
    assert ara_handoff["reproducibility"]["network_required"] is False
    assert ara_handoff["required_gate_inputs"]["run_ledger"] == "memory/run_ledger.jsonl"
    assert ara_handoff["required_gate_inputs"]["ablation_report"] == "artifacts/ablation_report.json"
    assert ara_handoff["required_gate_inputs"]["experiment_memory"] == "memory/experiment_memory.json"
    assert (output / "artifacts" / "reproducibility_notes.md").read_text(encoding="utf-8").strip()
    assert ablation_report["ablations"][0]["metrics"]["error_count"] == 6
    assert error_analysis["primary_error_count"] == 0
    assert experiment_memory["run_id"] == payload["run_id"]
    assert json.loads(experiment_memory_lines[0]) == experiment_memory
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
    assert result.metadata["artifact_count"] == 15
    assert result.metadata["claim_count"] == 1
    assert result.metadata["ara_gate"]["required_inputs_present"] is True


def test_local_benchmark_cli_emits_json(tmp_path, capsys):
    output = tmp_path / "cli_bundle"

    exit_code = cli.main(["run-local-benchmark", "--out", str(output), "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "passed"
    assert payload["benchmark"]["metrics"]["accuracy_delta"] == 0.5
    assert payload["analysis"]["ablation_error_count"] == 6
    assert payload["run_ledger"]["entry"]["status"] == "passed"
    assert payload["experiment_memory"]["entry"]["ablation_findings"][0]["error_count"] == 6


def test_local_benchmark_is_registered():
    payload = registry_payload()

    dataset_ids = {item["id"] for item in payload["datasets"]}
    model_ids = {item["id"] for item in payload["models"]}
    benchmarks = {item["id"]: item for item in payload["benchmarks"]}

    assert LOCAL_DATASET_ID in dataset_ids
    assert LOCAL_MODEL_ID in model_ids
    assert "deterministic-pass-prior-baseline-v1" in model_ids
    assert LOCAL_ABLATION_MODEL_ID in model_ids
    assert benchmarks[LOCAL_BENCHMARK_ID]["entrypoint"] == "python3 -m aira run-local-benchmark"
    assert benchmarks[LOCAL_BENCHMARK_ID]["emits_artifact_kinds"] == [
        "benchmark_report",
        "ablation_report",
        "error_analysis",
        "provenance",
        "run_ledger",
        "experiment_memory",
    ]


def test_agent_selects_safe_registered_local_experiment(tmp_path):
    selection = select_local_experiment()
    plan = build_agent_plan(tmp_path / "agent_bundle")

    assert selection["benchmark"]["id"] == LOCAL_BENCHMARK_ID
    assert selection["dataset"]["id"] == LOCAL_DATASET_ID
    assert selection["models"][0]["id"] == LOCAL_MODEL_ID
    assert plan["schema_version"] == "aira.agent_plan.v1"
    assert plan["selected_registry_entries"] == {
        "benchmark_id": LOCAL_BENCHMARK_ID,
        "dataset_id": LOCAL_DATASET_ID,
        "model_ids": [
            LOCAL_MODEL_ID,
            "deterministic-pass-prior-baseline-v1",
            LOCAL_ABLATION_MODEL_ID,
        ],
        "primary_model_id": LOCAL_MODEL_ID,
    }
    assert [step["phase"] for step in plan["steps"]] == ["plan", "act", "observe", "reflect"]
    assert plan["bounds"]["live_model_calls"] is False
    assert plan["bounds"]["network_required"] is False


def test_agent_smoke_emits_valid_bundle_with_reusable_memory(tmp_path):
    output = tmp_path / "agent_bundle"

    payload = run_agent_smoke(output)

    assert payload["status"] == "passed"
    assert payload["validation"]["valid"] is True
    assert [step["phase"] for step in payload["loop"]] == ["plan", "act", "observe", "reflect"]
    assert all(step["status"] == "completed" for step in payload["loop"])
    assert payload["selected_registry_entries"]["benchmark_id"] == LOCAL_BENCHMARK_ID
    assert payload["memory"]["entry"]["outcome"] == "accepted"
    assert payload["memory"]["entry"]["bundle_valid"] is True
    assert payload["memory"]["entry"]["metrics"]["accuracy_delta"] == 0.5
    assert payload["memory"]["entry"]["analysis"]["ablation_error_count"] == 6
    assert "negative-term-ablation" in payload["memory"]["entry"]["experiment_memory"]["retrieval_keys"]

    agent_memory = json.loads((output / "memory" / "agent_memory.json").read_text(encoding="utf-8"))
    agent_memory_lines = (output / "memory" / "agent_memory.jsonl").read_text(encoding="utf-8").splitlines()
    agent_trace = json.loads((output / "artifacts" / "agent_trace.json").read_text(encoding="utf-8"))
    claims = json.loads((output / "claims.json").read_text(encoding="utf-8"))
    validation = validate_bundle(output)

    assert validation.valid
    assert agent_memory["entries"] == [payload["memory"]["entry"]]
    assert json.loads(agent_memory_lines[0]) == payload["memory"]["entry"]
    assert agent_trace["reflection"]["outcome"] == "accepted"
    assert agent_trace["observation"]["analysis"]["primary_error_count"] == 0
    assert validation.metadata["ara_gate"]["profile"] == "ara-public-bundle-reproduction-gate.v1"
    assert "artifacts/reproducibility_notes.md" in validation.metadata["ara_gate"]["reproducibility_note_artifacts"]
    assert {claim["claim_id"] for claim in claims["claims"]} >= {
        "aira-local-benchmark-c1",
        "aira-agent-smoke-c1",
    }


def test_agent_smoke_cli_emits_json(tmp_path, capsys):
    output = tmp_path / "agent_cli_bundle"

    exit_code = cli.main(["agent", "smoke", "--out", str(output), "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "passed"
    assert payload["validation"]["valid"] is True
    assert payload["memory"]["entry"]["selected_registry_entries"]["benchmark_id"] == LOCAL_BENCHMARK_ID
