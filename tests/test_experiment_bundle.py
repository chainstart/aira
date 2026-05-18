import json
import shutil

from aira import cli
from aira.benchmark import evaluate_fixture_benchmark, write_fixture_bundle
from aira.bundles import validate_bundle


def test_fixture_benchmark_metrics_are_deterministic():
    payload = evaluate_fixture_benchmark()

    assert payload["deterministic"] is True
    assert payload["live_model_calls"] is False
    assert payload["metrics"] == {
        "accuracy": 0.833333,
        "baseline_accuracy": 0.5,
        "accuracy_delta": 0.333333,
    }


def test_committed_aira_result_bundle_fixture_validates():
    result = validate_bundle("tests/fixtures/aira_result_bundle")

    assert result.valid
    assert result.bundle_type == "aira_result_bundle"
    assert result.metadata["artifact_count"] == 5
    assert result.metadata["claim_count"] == 1


def test_bundle_validator_rejects_confirmed_claim_without_reproduction_artifact(tmp_path):
    bundle = tmp_path / "bundle"
    shutil.copytree("tests/fixtures/aira_result_bundle", bundle)
    claims_path = bundle / "claims.json"
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    claims["claims"][0]["supported_by"] = ["metrics_table"]
    claims_path.write_text(json.dumps(claims), encoding="utf-8")

    result = validate_bundle(bundle)

    assert not result.valid
    assert any("without a reproduction status artifact" in error for error in result.errors)


def test_run_fixture_benchmark_writes_valid_result_bundle(tmp_path):
    output = tmp_path / "aira_fixture_bundle"

    payload = write_fixture_bundle(output)

    assert payload["status"] == "passed"
    assert payload["validation"]["valid"] is True
    assert (output / "bundle_manifest.json").is_file()
    assert (output / "artifacts" / "benchmark_report.json").is_file()


def test_bundle_validate_cli_emits_json(capsys):
    exit_code = cli.main(["bundles", "validate", "tests/fixtures/aira_result_bundle", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "passed"
    assert payload["valid"] is True


def test_run_fixture_benchmark_cli_emits_bundle(tmp_path, capsys):
    output = tmp_path / "cli_bundle"

    exit_code = cli.main(["run-fixture-benchmark", "--out", str(output), "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "passed"
    assert payload["validation"]["valid"] is True
    assert payload["benchmark"]["metrics"]["accuracy_delta"] == 0.333333
