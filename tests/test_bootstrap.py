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
    payload = manifest.to_dict()
    assert payload["safety"]["network_policy"] == "unrestricted"
    assert payload["safety"]["live_model_calls"] is True
    assert payload["profiles"]["production-open"]["gpu_required"] is True
    assert payload["profiles"]["production-open"]["package_installation"] is True


def test_labs_inspect_cli_emits_manifest_json(capsys):
    exit_code = cli.main(["labs", "inspect", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "found"
    assert payload["entrypoints"]["agent_cli"] == ["python3 -m aira"]
    assert payload["bundle_types"] == ["aira_result_bundle"]
    assert payload["bundle_handoff_profiles"][0]["profile"] == "ara-public-bundle-reproduction-gate.v1"
    assert "artifacts/ara_handoff.json" in payload["bundle_handoff_profiles"][0]["required_artifacts"]
    assert any(profile["profile"] == "ara-production-open" for profile in payload["bundle_handoff_profiles"])
    assert payload["registries"]["datasets"] == "aira/registries/datasets.json"
    assert "python3 -m aira run-local-benchmark" in payload["entrypoints"]["direct_tools"]
    assert "python3 -m aira agent smoke" in payload["entrypoints"]["direct_tools"]
    assert "python3 -m aira experiments run --profile production-open" in payload["entrypoints"]["direct_tools"]


def test_registry_placeholders_are_local_and_deterministic():
    payload = registry_payload()

    assert payload["schema_version"] == "aira.registry.v1"
    assert payload["datasets"][0]["network_required"] is False
    assert all(model["live_model_calls"] is False for model in payload["models"])
    assert payload["benchmarks"][0]["emits_bundle_type"] == "aira_result_bundle"
    assert {dataset["id"] for dataset in payload["datasets"]} >= {
        "fixture-ai-classification",
        "local-experiment-outcomes-v1",
    }
    assert {benchmark["id"] for benchmark in payload["benchmarks"]} >= {
        "fixture-classification-smoke",
        "local-text-outcome-classification",
    }
