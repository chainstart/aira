import json

from aira import cli
from aira.bundles import validate_bundle
from aira.deepening import build_ara_deepening_plan, run_ara_deepening_experiment


def test_build_ara_deepening_plan_maps_task_gaps_to_artifacts():
    plan = build_ara_deepening_plan(
        "tests/fixtures/ara_deepening_task.json",
        source_bundle="tests/fixtures/aira_result_bundle",
        profile_name="production-open",
    )

    assert plan["schema_version"] == "aira.production_plan.v1"
    assert plan["plan_schema_version"] == "aira.ara_deepening_plan.v1"
    assert plan["network_required"] is True
    assert plan["claims"][0]["status"] == "confirmed"
    assert "primary_contribution" in plan["claims"][0]["supported_by"]
    assert "mechanism_insight" in plan["claims"][0]["supported_by"]
    assert "artifact_availability" in plan["claims"][0]["supported_by"]
    outputs = {item["artifact_id"]: item for item in plan["tasks"][0]["outputs"]}
    assert outputs["primary_contribution"]["kind"] == "primary_contribution"
    assert outputs["mechanism_insight"]["kind"] == "mechanism_insight"
    assert outputs["artifact_availability"]["kind"] == "artifact_availability"


def test_ara_deepening_experiment_emits_valid_production_open_bundle(tmp_path):
    output = tmp_path / "deepening_bundle"

    payload = run_ara_deepening_experiment(
        profile_name="production-open",
        task_package="tests/fixtures/ara_deepening_task.json",
        source_bundle="tests/fixtures/aira_result_bundle",
        output_dir=output,
    )

    assert payload["status"] == "passed"
    assert payload["profile"] == "production-open"
    assert payload["validation"]["valid"] is True

    validation = validate_bundle(output, profile="ara-production-open")
    assert validation.valid
    artifact_ids = {item["artifact_id"] for item in json.loads((output / "artifact_manifest.json").read_text())["artifacts"]}
    assert {
        "ara_deepening_task",
        "primary_contribution",
        "mechanism_insight",
        "artifact_availability",
        "top_venue_evidence",
        "deepening_report",
    } <= artifact_ids
    claims = json.loads((output / "claims.json").read_text())["claims"]
    assert claims[0]["evidence_level"] == "confirmed_with_reproduction"
    assert claims[0]["reproduction_status"] == "reproduced"
    assert "reproduction_status" in claims[0]["supported_by"]


def test_experiments_deepen_cli_emits_json(tmp_path, capsys):
    output = tmp_path / "cli_deepening_bundle"

    exit_code = cli.main(
        [
            "experiments",
            "deepen",
            "--profile",
            "production-open",
            "--task",
            "tests/fixtures/ara_deepening_task.json",
            "--source-bundle",
            "tests/fixtures/aira_result_bundle",
            "--out",
            str(output),
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "passed"
    assert payload["task_package"].endswith("tests/fixtures/ara_deepening_task.json")
    assert validate_bundle(output, profile="ara-production-open").valid
