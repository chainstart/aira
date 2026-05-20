import json

from aira import cli
from aira.agent import run_agent_smoke
from aira.benchmark import LOCAL_DATASET_ID, LOCAL_MODEL_ID, write_local_benchmark_bundle
from aira.memory import MEMORY_INDEX_SCHEMA_VERSION, build_memory_index


def test_memory_index_promotes_bundle_local_experiment_memory(tmp_path):
    bundle = tmp_path / "local_bundle"
    output = tmp_path / "memory_index"
    run_payload = write_local_benchmark_bundle(bundle)

    index = build_memory_index([bundle], output)

    assert index["schema_version"] == MEMORY_INDEX_SCHEMA_VERSION
    assert index["status"] == "passed"
    assert index["run_count"] == 1
    assert index["runs"][0]["run_id"] == run_payload["run_id"]
    assert index["runs"][0]["dataset_id"] == LOCAL_DATASET_ID
    assert index["runs"][0]["model_id"] == LOCAL_MODEL_ID
    assert "negative-term-ablation" in index["retrieval"]["keys"]
    assert index["failures"][0]["failure_kind"] == "ablation_regression"
    assert index["failures"][0]["count"] == 6
    assert "dataset_sha256" in index["fingerprints"]["by_run"][run_payload["run_id"]]
    assert index["outcomes"]["matrix"][0]["best_accuracy"] == 1.0

    persisted = json.loads((output / "memory_index.json").read_text(encoding="utf-8"))
    runs_jsonl = (output / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    failures_jsonl = (output / "failures.jsonl").read_text(encoding="utf-8").splitlines()
    assert persisted["run_count"] == 1
    assert json.loads(runs_jsonl[0])["run_id"] == run_payload["run_id"]
    assert json.loads(failures_jsonl[0])["failure_kind"] == "ablation_regression"


def test_memory_index_retrieves_agent_reflections(tmp_path):
    bundle = tmp_path / "agent_bundle"
    output = tmp_path / "memory_index"
    payload = run_agent_smoke(bundle)

    index = build_memory_index([bundle], output)

    assert index["run_count"] == 1
    assert index["reflection_count"] == 1
    reflection = index["retrieval"]["reflections"][0]
    assert reflection["run_id"] == payload["run_id"]
    assert reflection["outcome"] == "accepted"
    assert "Require bundle validation" in reflection["retrieval_text"]
    assert index["runs"][0]["agent_memory"]["outcome"] == "accepted"


def test_memory_index_lifecycle_filters_and_cli_json(tmp_path, capsys):
    passed_bundle = tmp_path / "passed_bundle"
    filtered_output = tmp_path / "filtered_memory"
    cli_output = tmp_path / "cli_memory"
    write_local_benchmark_bundle(passed_bundle)

    filtered = build_memory_index([passed_bundle], filtered_output, status_filter="failed", max_runs=1)

    assert filtered["run_count"] == 0
    assert filtered["lifecycle"]["status_filter"] == "failed"
    assert filtered["lifecycle"]["retention_policy"] == "latest_1_matching_runs"

    exit_code = cli.main(
        [
            "memory",
            "index",
            "--runs",
            str(passed_bundle),
            "--out",
            str(cli_output),
            "--max-runs",
            "1",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "passed"
    assert payload["run_count"] == 1
    assert payload["artifacts"]["index"] == "memory_index.json"
