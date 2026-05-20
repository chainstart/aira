import json

from aira import cli
from aira.bundles import validate_bundle
from aira.production_evaluation import evaluate_production_bundle
from aira.production_runner import run_production_experiment


def test_production_evaluation_appends_machine_readable_bundle_artifacts(tmp_path):
    output = tmp_path / "prod_bundle"
    run_production_experiment("production-local", "tests/fixtures/production_plan.json", output)

    payload = evaluate_production_bundle(output)

    assert payload["status"] == "passed"
    assert payload["metrics"]["schema_version"] == "aira.production_evaluation_metrics.v1"
    assert payload["metrics"]["row_count"] == 4
    assert payload["metrics"]["metrics"] == {
        "accuracy": 1.0,
        "macro_f1": 1.0,
        "baseline_accuracy": 0.5,
        "baseline_macro_f1": 0.333333,
        "accuracy_delta_vs_baseline": 0.5,
    }
    assert payload["ablation_matrix"]["rows"][0]["metrics"] == {
        "accuracy": 0.5,
        "macro_f1": 0.333333,
        "accuracy_delta_vs_primary": -0.5,
        "changed_prediction_count": 2,
        "error_count": 2,
    }
    assert payload["error_taxonomy"]["primary_error_count"] == 0
    assert payload["error_taxonomy"]["ablation_error_count"] == 2
    assert payload["error_taxonomy"]["taxonomy"] == [
        {
            "count": 2,
            "description": "A failure-labeled example was predicted as pass after failure keyword behavior was disabled.",
            "error_type": "false_pass_without_failure_terms",
        }
    ]
    assert payload["statistical_tests"]["tests"][0]["p_value"] == 0.5
    assert payload["statistical_tests"]["effect_sizes"]["accuracy_delta_primary_vs_ablation"] == 0.5
    assert payload["report_summary"]["summary"]["mcnemar_significant"] is False
    assert payload["validation"]["valid"] is True

    artifact_manifest = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
    artifact_ids = {artifact["artifact_id"] for artifact in artifact_manifest["artifacts"]}
    assert artifact_ids >= {
        "production_evaluation_metrics",
        "production_ablation_matrix",
        "production_error_taxonomy",
        "production_statistical_tests",
        "production_report_summary",
    }
    claims = json.loads((output / "claims.json").read_text(encoding="utf-8"))
    assert "aira-production-evaluation-c1" in {claim["claim_id"] for claim in claims["claims"]}
    manifest = json.loads((output / "bundle_manifest.json").read_text(encoding="utf-8"))
    assert manifest["production_evaluation"]["status"] == "passed"
    assert validate_bundle(output).valid


def test_production_evaluation_is_reproducible(tmp_path):
    first = tmp_path / "first_bundle"
    second = tmp_path / "second_bundle"
    run_production_experiment("production-local", "tests/fixtures/production_plan.json", first)
    run_production_experiment("production-local", "tests/fixtures/production_plan.json", second)

    first_payload = evaluate_production_bundle(first)
    second_payload = evaluate_production_bundle(second)

    assert first_payload["metrics"]["metrics"] == second_payload["metrics"]["metrics"]
    assert first_payload["ablation_matrix"]["rows"] == second_payload["ablation_matrix"]["rows"]
    assert first_payload["error_taxonomy"]["taxonomy"] == second_payload["error_taxonomy"]["taxonomy"]
    assert first_payload["statistical_tests"]["tests"] == second_payload["statistical_tests"]["tests"]


def test_production_evaluation_cli_emits_json(tmp_path, capsys):
    output = tmp_path / "prod_cli_bundle"
    run_production_experiment("production-local", "tests/fixtures/production_plan.json", output)

    exit_code = cli.main(["experiments", "evaluate", "--bundle", str(output), "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "passed"
    assert payload["metrics"]["metrics"]["accuracy"] == 1.0
    assert payload["ablation_matrix"]["rows"][0]["metrics"]["error_count"] == 2
    assert payload["validation"]["valid"] is True
