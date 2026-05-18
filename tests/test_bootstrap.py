import json

from aira import cli
from aira.manifest import load_manifest
from aira.registries import registry_payload


def test_research_lab_manifest_is_valid():
    manifest = load_manifest("research_lab.yaml")

    assert manifest.validation.valid
    assert manifest.lab_id == "aira"
    assert manifest.domain == "ai_ml"
    assert manifest.bundle_types == ["aira_result_bundle"]
    assert manifest.to_dict()["safety"]["live_model_calls"] is False


def test_labs_inspect_cli_emits_manifest_json(capsys):
    exit_code = cli.main(["labs", "inspect", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "found"
    assert payload["entrypoints"]["agent_cli"] == ["python3 -m aira"]
    assert payload["bundle_types"] == ["aira_result_bundle"]
    assert payload["registries"]["datasets"] == "aira/registries/datasets.json"


def test_registry_placeholders_are_local_and_deterministic():
    payload = registry_payload()

    assert payload["schema_version"] == "aira.registry.v1"
    assert payload["datasets"][0]["network_required"] is False
    assert all(model["live_model_calls"] is False for model in payload["models"])
    assert payload["benchmarks"][0]["emits_bundle_type"] == "aira_result_bundle"
