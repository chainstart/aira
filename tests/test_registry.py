import json

from aira import cli
from aira.registries import registry_payload


def test_default_registry_view_preserves_fixture_and_local_entries():
    payload = registry_payload()

    assert payload["schema_version"] == "aira.registry.v1"
    assert payload["profile"] == "default"
    assert [dataset["id"] for dataset in payload["datasets"]] == [
        "fixture-ai-classification",
        "local-experiment-outcomes-v1",
    ]
    assert "production-local-controlled-python-runner-v1" not in {
        model["id"] for model in payload["models"]
    }
    assert all(dataset.get("network_required") is False for dataset in payload["datasets"])
    assert all(model["live_model_calls"] is False for model in payload["models"])


def test_registries_cli_keeps_default_payload(capsys):
    exit_code = cli.main(["registries", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "available"
    assert payload["profile"] == "default"
    assert {benchmark["id"] for benchmark in payload["benchmarks"]} == {
        "fixture-classification-smoke",
        "local-text-outcome-classification",
    }


def test_registries_cli_can_print_production_profile(capsys):
    exit_code = cli.main(["registries", "--profile", "production-local", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile"] == "production-local"
    assert "operator-supplied-production-plan" in {dataset["id"] for dataset in payload["datasets"]}
    assert "production-local-plan-execution" in {benchmark["id"] for benchmark in payload["benchmarks"]}
