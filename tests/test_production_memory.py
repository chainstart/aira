import json

from aira import cli
from aira.memory import build_memory_index
from aira.production_evaluation import evaluate_production_bundle
from aira.production_runner import PLAN_SCHEMA_VERSION, run_production_experiment


def test_production_memory_indexes_model_dataset_outcomes_and_failures(tmp_path):
    bundle = tmp_path / "prod_bundle"
    output = tmp_path / "prod_memory"
    run_payload = run_production_experiment("production-local", "tests/fixtures/production_plan.json", bundle)
    evaluate_production_bundle(bundle)

    index = build_memory_index([bundle], output)

    assert index["status"] == "passed"
    assert index["run_count"] == 1
    run = index["runs"][0]
    provenance = json.loads((bundle / "artifacts" / "provenance.json").read_text(encoding="utf-8"))

    assert run["run_id"] == run_payload["run_id"]
    assert run["status"] == "passed"
    assert run["metrics"]["passed_task_count"] == 2
    assert run["evaluation"]["metrics"]["accuracy"] == 1.0
    assert run["evaluation"]["ablation_error_count"] == 2
    assert run["input_fingerprints"]["dataset_sha256"] == provenance["input_fingerprints"]["dataset_sha256"]

    matrix = index["outcomes"]["matrix"][0]
    assert matrix["dataset_id"] == run["dataset_id"]
    assert matrix["model_id"] == run["model_id"]
    assert matrix["passed_count"] == 1
    assert matrix["best_accuracy"] == 1.0
    assert index["fingerprints"]["by_fingerprint"]["dataset_sha256"][run["input_fingerprints"]["dataset_sha256"]] == [
        run["run_id"]
    ]

    failure = next(item for item in index["failures"] if item["failure_kind"] == "evaluation_error_taxonomy")
    assert failure["error_type"] == "false_pass_without_failure_terms"
    assert failure["count"] == 2


def test_production_memory_indexes_failed_task_ledger(tmp_path):
    plan = tmp_path / "failed_plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": PLAN_SCHEMA_VERSION,
                "plan_id": "memory-failed-plan",
                "tasks": [
                    {
                        "id": "fails",
                        "command": {"kind": "inline_python", "code": "raise SystemExit(7)\n"},
                        "outputs": [{"artifact_id": "missing", "path": "missing.txt"}],
                    },
                    {
                        "id": "dependent",
                        "dependencies": ["fails"],
                        "command": {
                            "kind": "inline_python",
                            "code": "from pathlib import Path\nPath('out.txt').write_text('bad')\n",
                        },
                        "outputs": [{"artifact_id": "dependent_output", "path": "out.txt"}],
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    bundle = tmp_path / "failed_bundle"
    output = tmp_path / "memory"
    run_production_experiment("production-local", plan, bundle)

    index = build_memory_index([bundle], output, status_filter="failed")

    assert index["run_count"] == 1
    assert index["runs"][0]["status"] == "failed"
    failure_kinds = {failure["failure_kind"] for failure in index["failures"]}
    assert failure_kinds >= {"run_failed", "task_failed", "task_skipped"}


def test_production_memory_cli_writes_reusable_index_files(tmp_path, capsys):
    bundle = tmp_path / "prod_bundle"
    output = tmp_path / "prod_memory"
    run_production_experiment("production-local", "tests/fixtures/production_plan.json", bundle)

    exit_code = cli.main(["memory", "index", "--runs", str(bundle), "--out", str(output), "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_count"] == 1
    assert payload["failure_count"] == 0
    assert (output / "memory_index.json").exists()
    assert (output / "fingerprints.json").exists()
    assert (output / "outcomes.json").exists()
    assert (output / "reflections.json").exists()
