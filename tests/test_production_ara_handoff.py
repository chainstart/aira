import json

from aira import cli
from aira.agent import run_agent_smoke, run_production_agent_smoke
from aira.bundles import validate_bundle
from aira.manifest import load_manifest


def test_production_agent_smoke_emits_ara_ready_bundle(tmp_path):
    output = tmp_path / "prod_ara_bundle"

    payload = run_production_agent_smoke(output)

    assert payload["status"] == "passed"
    assert payload["profile"] == "production-local"
    assert payload["validation_profile"] == "ara-production"
    assert payload["validation"]["valid"] is True

    manifest = json.loads((output / "bundle_manifest.json").read_text(encoding="utf-8"))
    handoff = json.loads((output / "artifacts" / "ara_handoff.json").read_text(encoding="utf-8"))
    artifact_manifest = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
    artifact_ids = {artifact["artifact_id"] for artifact in artifact_manifest["artifacts"]}

    assert manifest["ara_handoff"]["validation_profile"] == "ara-production"
    assert manifest["production_runner"]["profile"] == "production-local"
    assert manifest["production_evaluation"]["status"] == "passed"
    assert handoff["dispatch"]["allowed_interfaces"] == ["research_lab.yaml", "aira_result_bundle"]
    assert handoff["dispatch"]["validation_command"] == (
        "python3 -m aira bundles validate <bundle> --profile ara-production --json"
    )
    assert {
        "ara_handoff",
        "reproducibility_notes",
        "production_evaluation_metrics",
        "production_memory_index",
        "production_memory_runs",
        "production_memory_failures",
        "production_memory_fingerprints",
        "production_memory_outcomes",
        "production_memory_reflections",
    } <= artifact_ids

    validation = validate_bundle(output, profile="ara-production")
    assert validation.valid
    assert validation.metadata["validation_profile"] == "ara-production"
    assert validation.metadata["ara_gate"]["required_inputs_present"] is True


def test_production_ara_validation_profile_rejects_plain_local_agent_bundle(tmp_path):
    output = tmp_path / "local_agent_bundle"
    local_payload = run_agent_smoke(output)

    validation = validate_bundle(output, profile="ara-production")

    assert local_payload["status"] == "passed"
    assert not validation.valid
    assert any("production_runner" in error for error in validation.errors)


def test_production_handoff_cli_and_manifest_dispatch(tmp_path, capsys):
    output = tmp_path / "prod_cli_bundle"

    exit_code = cli.main(["agent", "production-smoke", "--out", str(output), "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "passed"
    assert payload["validation"]["metadata"]["validation_profile"] == "ara-production"

    exit_code = cli.main(["bundles", "validate", str(output), "--profile", "ara-production", "--json"])

    assert exit_code == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["status"] == "passed"
    assert validation["metadata"]["validation_profile"] == "ara-production"

    manifest = load_manifest("research_lab.yaml").to_dict()
    production_profiles = [
        profile for profile in manifest["bundle_handoff_profiles"] if profile.get("profile") == "ara-production"
    ]
    production_open_profiles = [
        profile for profile in manifest["bundle_handoff_profiles"] if profile.get("profile") == "ara-production-open"
    ]
    assert production_profiles
    assert production_profiles[0]["dispatch"]["allowed_interfaces"] == [
        "research_lab.yaml",
        "aira_result_bundle",
    ]
    assert production_profiles[0]["validation_command"].endswith("--profile ara-production --json")
    assert production_open_profiles
    assert production_open_profiles[0]["dispatch"]["profile"] == "production-open"
    assert production_open_profiles[0]["validation_command"].endswith("--profile ara-production-open --json")
