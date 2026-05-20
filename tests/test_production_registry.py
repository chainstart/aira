import json

from aira import cli
from aira.registries import audit_registry, production_registry_payload


def _production_entries(payload, key, profile="production-local"):
    return [entry for entry in payload[key] if entry.get("profile") == profile]


def test_production_registry_includes_required_adapter_classes():
    payload = production_registry_payload()

    adapter_types = {
        entry["adapter"]["type"]
        for key in ("datasets", "models", "benchmarks")
        for entry in _production_entries(payload, key)
        if "adapter" in entry
    }
    assert adapter_types >= {
        "local_cache",
        "operator_supplied_artifact",
        "optional_external",
        "production_local_runner",
    }


def test_production_entries_record_fingerprint_version_policy_and_reproducibility():
    payload = production_registry_payload()

    for key in ("datasets", "models", "benchmarks"):
        for entry in _production_entries(payload, key):
            assert entry["version"]
            assert entry["fingerprint"]["algorithm"] == "sha256"
            assert len(entry["fingerprint"]["value"]) == 64
            assert "license_policy" in entry
            assert "resource_policy" in entry
            assert entry["reproducibility_notes"]
            assert entry.get("network_required") is False
            assert entry.get("external_datasets_required") is not True
            assert entry.get("gpu_required") is not True
            assert entry.get("live_model_calls") is not True


def test_optional_external_adapters_are_registered_but_disabled():
    payload = production_registry_payload()
    external_entries = [
        entry
        for key in ("datasets", "models")
        for entry in _production_entries(payload, key)
        if entry["adapter"]["type"] == "optional_external"
    ]

    assert external_entries
    assert all(entry["adapter"]["enabled"] is False for entry in external_entries)
    assert all(entry["adapter"]["network_required"] is False for entry in external_entries)


def test_production_registry_audit_passes_and_checks_references():
    audit = audit_registry("production-local")

    assert audit["status"] == "passed"
    assert audit["valid"] is True
    assert audit["errors"] == []
    assert audit["counts"]["production_datasets"] >= 3
    assert audit["counts"]["production_models"] >= 4
    assert audit["counts"]["production_benchmarks"] >= 1
    assert "registry_sha256" in audit
    assert any(check["id"] == "benchmark:production-local-plan-execution:references" for check in audit["checks"])


def test_production_open_registry_exposes_enabled_external_gpu_and_live_model_surface():
    payload = production_registry_payload("production-open")

    adapter_types = {
        entry["adapter"]["type"]
        for key in ("datasets", "models", "benchmarks")
        for entry in _production_entries(payload, key, "production-open")
        if "adapter" in entry
    }
    assert adapter_types >= {
        "optional_external",
        "builtin_runner",
        "hosted_model_api",
        "production_open_runner",
    }
    assert any(entry.get("network_required") is True for entry in _production_entries(payload, "datasets", "production-open"))
    assert any(entry.get("gpu_required") is True for entry in _production_entries(payload, "models", "production-open"))
    assert any(entry.get("live_model_calls") is True for entry in _production_entries(payload, "models", "production-open"))


def test_production_open_registry_audit_passes_and_checks_references():
    audit = audit_registry("production-open")

    assert audit["status"] == "passed"
    assert audit["valid"] is True
    assert audit["errors"] == []
    assert audit["counts"]["production_datasets"] >= 1
    assert audit["counts"]["production_models"] >= 2
    assert audit["counts"]["production_benchmarks"] >= 1
    assert "registry_sha256" in audit
    assert any(check["id"] == "benchmark:production-open-plan-execution:references" for check in audit["checks"])


def test_registry_audit_cli_emits_json(capsys):
    exit_code = cli.main(["registry", "audit", "--profile", "production-local", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "passed"
    assert payload["profile"] == "production-local"
    assert "local_cache" in payload["adapter_types"]


def test_registry_audit_cli_emits_production_open_json(capsys):
    exit_code = cli.main(["registry", "audit", "--profile", "production-open", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "passed"
    assert payload["profile"] == "production-open"
    assert "hosted_model_api" in payload["adapter_types"]
