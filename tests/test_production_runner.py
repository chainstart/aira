import json

from aira import cli
from aira.bundles import validate_bundle
from aira.production_runner import (
    PLAN_SCHEMA_VERSION,
    evaluate_production_policy,
    load_profile,
    run_production_experiment,
)


def test_production_local_runner_emits_valid_bundle(tmp_path):
    output = tmp_path / "prod_bundle"
    plan = "tests/fixtures/production_plan.json"

    payload = run_production_experiment("production-local", plan, output)

    assert payload["status"] == "passed"
    assert payload["profile"] == "production-local"
    assert payload["policy"]["allowed"] is True
    assert [task["status"] for task in payload["tasks"]] == ["passed", "passed"]
    assert {artifact["artifact_id"] for artifact in payload["materialized_artifacts"]} == {
        "production_dataset",
        "prepare_results",
        "production_metrics",
        "production_predictions",
    }
    assert payload["validation"]["valid"] is True

    validation = validate_bundle(output)
    manifest = json.loads((output / "bundle_manifest.json").read_text(encoding="utf-8"))
    policy = json.loads((output / "artifacts" / "policy_report.json").read_text(encoding="utf-8"))
    trace = json.loads((output / "artifacts" / "execution_trace.json").read_text(encoding="utf-8"))
    ledger = json.loads((output / "artifacts" / "run_ledger_entry.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "artifacts" / "tasks" / "score_dataset" / "metrics.json").read_text(encoding="utf-8"))

    assert validation.valid
    assert manifest["production_runner"]["profile"] == "production-local"
    assert manifest["network_required"] is False
    assert policy["allowed"] is True
    assert trace["profile"]["network_policy"] == "none"
    assert ledger["status"] == "passed"
    assert ledger["metrics"]["passed_task_count"] == 2
    assert metrics == {"accuracy": 1.0, "row_count": 4, "success": True}


def test_production_open_runner_emits_valid_open_bundle(tmp_path):
    output = tmp_path / "prod_open_bundle"
    plan = "tests/fixtures/production_open_plan.json"

    payload = run_production_experiment("production-open", plan, output)

    assert payload["status"] == "passed"
    assert payload["profile"] == "production-open"
    assert payload["policy"]["allowed"] is True
    assert [task["status"] for task in payload["tasks"]] == ["passed", "passed"]
    assert [task["command_kind"] for task in payload["tasks"]] == ["inline_python", "external_command"]
    assert payload["validation"]["valid"] is True

    validation = validate_bundle(output, profile="ara-production-open")
    manifest = json.loads((output / "bundle_manifest.json").read_text(encoding="utf-8"))
    policy = json.loads((output / "artifacts" / "policy_report.json").read_text(encoding="utf-8"))
    trace = json.loads((output / "artifacts" / "execution_trace.json").read_text(encoding="utf-8"))
    open_report = json.loads(
        (output / "artifacts" / "tasks" / "open_capability_report" / "open_capability_report.json").read_text(
            encoding="utf-8"
        )
    )
    external_report = (
        output / "artifacts" / "tasks" / "external_command_report" / "external_command_report.txt"
    ).read_text(encoding="utf-8")

    assert validation.valid
    assert validation.metadata["validation_profile"] == "ara-production-open"
    assert manifest["production_runner"]["profile"] == "production-open"
    assert manifest["model_id"] == "production-open-python-runner-v1"
    assert manifest["deterministic"] is False
    assert manifest["network_required"] is True
    assert manifest["external_datasets_required"] is True
    assert manifest["gpu_required"] is True
    assert manifest["live_model_calls"] is True
    assert policy["allowed"] is True
    assert trace["profile"]["network_policy"] == "unrestricted"
    assert open_report["profile"] == "production-open"
    assert open_report["network_policy"] == "unrestricted"
    assert open_report["urllib_available"] is True
    assert open_report["subprocess_available"] is True
    assert external_report == "external command executed\n"


def test_production_runner_cli_emits_json(tmp_path, capsys):
    output = tmp_path / "prod_cli_bundle"

    exit_code = cli.main(
        [
            "experiments",
            "run",
            "--profile",
            "production-local",
            "--plan",
            "tests/fixtures/production_plan.json",
            "--out",
            str(output),
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "passed"
    assert payload["policy"]["allowed"] is True
    assert payload["validation"]["valid"] is True


def test_production_profile_rejects_implicit_or_unknown_profiles():
    try:
        load_profile("default")
    except ValueError as exc:
        assert "production-local" in str(exc)
    else:
        raise AssertionError("unknown profile should be rejected")


def test_production_policy_blocks_packages_and_dangerous_imports():
    profile = load_profile("production-local")
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_id": "blocked-plan",
        "resource_requirements": {"python_packages": ["requests"]},
        "tasks": [
            {
                "id": "unsafe",
                "command": {"kind": "inline_python", "code": "import subprocess\nprint('no')\n"},
                "outputs": [{"artifact_id": "unsafe_output", "path": "out.txt"}],
            }
        ],
    }

    policy, task_order = evaluate_production_policy(plan, profile)

    assert task_order == ["unsafe"]
    assert policy.allowed is False
    assert any("Package is not allowed" in error for error in policy.errors)
    assert any("denied modules" in error for error in policy.errors)


def test_production_runner_isolates_task_failures_and_skips_dependents(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": PLAN_SCHEMA_VERSION,
                "plan_id": "failure-isolation-plan",
                "tasks": [
                    {
                        "id": "fails",
                        "command": {"kind": "inline_python", "code": "raise SystemExit(7)\n"},
                        "outputs": [{"artifact_id": "failed_output", "path": "missing.txt"}],
                    },
                    {
                        "id": "dependent",
                        "dependencies": ["fails"],
                        "command": {"kind": "inline_python", "code": "from pathlib import Path\nPath('out.txt').write_text('bad')\n"},
                        "outputs": [{"artifact_id": "dependent_output", "path": "out.txt"}],
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    payload = run_production_experiment("production-local", plan, tmp_path / "failed_bundle")

    assert payload["status"] == "failed"
    assert [task["status"] for task in payload["tasks"]] == ["failed", "skipped"]
    assert payload["validation"]["valid"] is True
    assert payload["materialized_artifacts"] == []
