"""Registries for AIRA datasets, models, and benchmarks."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any


REGISTRY_SCHEMA_VERSION = "aira.registry.v1"
PRODUCTION_PROFILE = "production-local"
PRODUCTION_OPEN_PROFILE = "production-open"
PRODUCTION_PROFILES = {PRODUCTION_PROFILE, PRODUCTION_OPEN_PROFILE}
PRODUCTION_REGISTRY_SCHEMA_VERSION = "aira.production_registry.v1"


def _fingerprint(kind: str, entry_id: str, version: str) -> dict[str, str]:
    value = hashlib.sha256(f"{kind}:{entry_id}:{version}".encode("utf-8")).hexdigest()
    return {
        "algorithm": "sha256",
        "value": value,
        "material": f"{kind}:{entry_id}:{version}",
    }


DATASETS: list[dict[str, Any]] = [
    {
        "id": "fixture-ai-classification",
        "name": "Deterministic fixture classification set",
        "status": "placeholder",
        "source": "local_synthetic",
        "rows": 6,
        "network_required": False,
        "intended_use": "Smoke-test bundle emission and validator contracts.",
    },
    {
        "id": "local-experiment-outcomes-v1",
        "name": "Local experiment outcome text classification set",
        "status": "local_deterministic",
        "source": "builtin_local_fixture",
        "rows": 12,
        "label_set": ["fail", "pass"],
        "splits": ["core", "handoff"],
        "network_required": False,
        "external_datasets_required": False,
        "intended_use": "Exercise local benchmark execution, provenance, ablation analysis, and run memory.",
    },
]


PRODUCTION_DATASETS: list[dict[str, Any]] = [
    {
        "id": "operator-supplied-production-plan",
        "name": "Operator supplied production-local plan artifacts",
        "status": "production_local",
        "source": "operator_supplied_artifact",
        "profile": PRODUCTION_PROFILE,
        "version": "1.0.0",
        "versioning": {
            "scheme": "semver",
            "stability": "operator controlled",
            "change_policy": "Any task, artifact, schema, or label change requires a new plan fingerprint.",
        },
        "adapter": {
            "type": "operator_supplied_artifact",
            "enabled": True,
            "network_required": False,
            "accepted_artifact_kinds": ["production_plan", "dataset", "task_output"],
            "path_policy": "relative_bundle_paths_only",
        },
        "fingerprint": _fingerprint("dataset", "operator-supplied-production-plan", "1.0.0"),
        "license_policy": {
            "default_license": "operator_supplied",
            "requires_operator_attestation": True,
            "redistribution_allowed": "operator_defined",
        },
        "resource_policy": {
            "storage": "bundle_local",
            "network_required": False,
            "external_datasets_required": False,
            "gpu_required": False,
            "max_rows": "operator_declared",
        },
        "reproducibility_notes": [
            "The production runner records the source plan and per-output sha256 values in the result bundle.",
            "Operators are responsible for preserving the exact supplied artifacts and license attestations.",
        ],
        "network_required": False,
        "external_datasets_required": False,
        "intended_use": "Represent datasets and task outputs supplied through an explicit production-local plan.",
    },
    {
        "id": "production-local-cache-dataset-v1",
        "name": "Production-local dataset cache adapter",
        "status": "production_local_adapter",
        "source": "local_cache",
        "profile": PRODUCTION_PROFILE,
        "version": "1.0.0",
        "versioning": {
            "scheme": "content_addressed",
            "stability": "sha256 pinned",
            "change_policy": "Cache entries are immutable by fingerprint; refreshes create new fingerprints.",
        },
        "adapter": {
            "type": "local_cache",
            "enabled": True,
            "network_required": False,
            "cache_root": ".aira/cache/datasets",
            "lookup_keys": ["id", "version", "sha256"],
        },
        "fingerprint": _fingerprint("dataset", "production-local-cache-dataset-v1", "1.0.0"),
        "license_policy": {
            "default_license": "operator_supplied",
            "requires_operator_attestation": True,
            "redistribution_allowed": "operator_defined",
        },
        "resource_policy": {
            "storage": "local_cache",
            "network_required": False,
            "external_datasets_required": False,
            "gpu_required": False,
            "max_size_mb": 512,
        },
        "reproducibility_notes": [
            "Cache hits must be matched by sha256 before use.",
            "The cache is optional; bundles remain reproducible from materialized artifacts.",
        ],
        "network_required": False,
        "external_datasets_required": False,
        "intended_use": "Reuse operator-approved local dataset artifacts without network access.",
    },
    {
        "id": "optional-external-dataset-adapter-template-v1",
        "name": "Optional external dataset adapter template",
        "status": "disabled_template",
        "source": "external_adapter",
        "profile": PRODUCTION_PROFILE,
        "version": "0.1.0",
        "versioning": {
            "scheme": "adapter_contract",
            "stability": "disabled until explicitly implemented",
            "change_policy": "Enabling requires a new profile, policy review, fingerprint, and license metadata.",
        },
        "adapter": {
            "type": "optional_external",
            "enabled": False,
            "network_required": False,
            "provider": "placeholder",
            "activation_policy": "not_enabled_for_production_local",
        },
        "fingerprint": _fingerprint("dataset", "optional-external-dataset-adapter-template-v1", "0.1.0"),
        "license_policy": {
            "default_license": "unknown_until_operator_attested",
            "requires_operator_attestation": True,
            "redistribution_allowed": False,
        },
        "resource_policy": {
            "storage": "none_until_enabled",
            "network_required": False,
            "external_datasets_required": False,
            "gpu_required": False,
        },
        "reproducibility_notes": [
            "This is a registry contract placeholder only; production-local audit verifies it is disabled.",
            "No external download is performed by AIRA under the production-local profile.",
        ],
        "network_required": False,
        "external_datasets_required": False,
        "intended_use": "Document the external-adapter contract without enabling external datasets.",
    },
]


MODELS: list[dict[str, Any]] = [
    {
        "id": "fixture-threshold-classifier",
        "name": "Fixture threshold classifier",
        "status": "placeholder",
        "implementation": "aira.benchmark.threshold_predict",
        "live_model_calls": False,
        "intended_use": "Deterministic benchmark smoke only.",
    },
    {
        "id": "fixture-majority-baseline",
        "name": "Fixture majority baseline",
        "status": "placeholder",
        "implementation": "aira.benchmark.majority_predict",
        "live_model_calls": False,
        "intended_use": "Local comparison baseline for fixture smoke.",
    },
    {
        "id": "deterministic-keyword-outcome-classifier-v1",
        "name": "Deterministic keyword outcome classifier",
        "status": "local_deterministic",
        "implementation": "aira.benchmark.keyword_outcome_predict",
        "live_model_calls": False,
        "network_required": False,
        "gpu_required": False,
        "intended_use": "Local text classification benchmark runner without external dependencies.",
    },
    {
        "id": "deterministic-pass-prior-baseline-v1",
        "name": "Deterministic pass-prior baseline",
        "status": "local_deterministic",
        "implementation": "aira.benchmark.pass_prior_predict",
        "live_model_calls": False,
        "network_required": False,
        "gpu_required": False,
        "intended_use": "Deterministic baseline for the local text outcome benchmark.",
    },
    {
        "id": "deterministic-keyword-no-negative-ablation-v1",
        "name": "Deterministic keyword classifier without negative terms",
        "status": "local_deterministic_ablation",
        "implementation": "aira.benchmark.keyword_no_negative_predict",
        "live_model_calls": False,
        "network_required": False,
        "gpu_required": False,
        "intended_use": "Local ablation fixture proving negative outcome terms are required for fail examples.",
    },
]


PRODUCTION_MODELS: list[dict[str, Any]] = [
    {
        "id": "production-local-controlled-python-runner-v1",
        "name": "Production-local controlled Python runner",
        "status": "production_local",
        "profile": PRODUCTION_PROFILE,
        "version": "1.0.0",
        "versioning": {
            "scheme": "semver",
            "stability": "profile pinned",
            "change_policy": "Policy, import deny-list, or resource limit changes require a new fingerprint.",
        },
        "implementation": "aira.production_runner.run_production_experiment",
        "adapter": {
            "type": "builtin_runner",
            "enabled": True,
            "network_required": False,
            "live_model_calls": False,
        },
        "fingerprint": _fingerprint("model", "production-local-controlled-python-runner-v1", "1.0.0"),
        "license_policy": {
            "default_license": "project_license",
            "requires_operator_attestation": False,
            "redistribution_allowed": True,
        },
        "resource_policy": {
            "network_required": False,
            "gpu_required": False,
            "live_model_calls": False,
            "package_installation": False,
            "max_cpu_threads": 2,
            "max_task_timeout_seconds": 30,
        },
        "reproducibility_notes": [
            "The runner executes local inline Python under the production-local policy profile.",
            "Model behavior is the policy-bounded execution contract, not a learned external model.",
        ],
        "live_model_calls": False,
        "network_required": False,
        "gpu_required": False,
        "intended_use": "Execute explicit production-local experiment plans without live model APIs.",
    },
    {
        "id": "operator-supplied-model-artifact-v1",
        "name": "Operator supplied model artifact adapter",
        "status": "production_local_adapter",
        "profile": PRODUCTION_PROFILE,
        "version": "1.0.0",
        "versioning": {
            "scheme": "content_addressed",
            "stability": "sha256 pinned",
            "change_policy": "Model artifact bytes, config, or runtime metadata changes require a new fingerprint.",
        },
        "implementation": "operator_supplied_artifact",
        "adapter": {
            "type": "operator_supplied_artifact",
            "enabled": True,
            "network_required": False,
            "accepted_artifact_kinds": ["model_config", "weights", "tokenizer", "inference_script"],
        },
        "fingerprint": _fingerprint("model", "operator-supplied-model-artifact-v1", "1.0.0"),
        "license_policy": {
            "default_license": "operator_supplied",
            "requires_operator_attestation": True,
            "redistribution_allowed": "operator_defined",
        },
        "resource_policy": {
            "storage": "bundle_or_local_cache",
            "network_required": False,
            "gpu_required": False,
            "live_model_calls": False,
            "package_installation": False,
            "max_size_mb": 1024,
        },
        "reproducibility_notes": [
            "Operators must provide artifact sha256 values and compatible runtime notes.",
            "AIRA does not fetch weights or call hosted models under production-local.",
        ],
        "live_model_calls": False,
        "network_required": False,
        "gpu_required": False,
        "intended_use": "Describe local model artifacts supplied by an operator for production-local experiments.",
    },
    {
        "id": "production-local-model-cache-v1",
        "name": "Production-local model cache adapter",
        "status": "production_local_adapter",
        "profile": PRODUCTION_PROFILE,
        "version": "1.0.0",
        "versioning": {
            "scheme": "content_addressed",
            "stability": "sha256 pinned",
            "change_policy": "Cache entries are immutable by model artifact fingerprint.",
        },
        "implementation": "local_cache",
        "adapter": {
            "type": "local_cache",
            "enabled": True,
            "network_required": False,
            "cache_root": ".aira/cache/models",
            "lookup_keys": ["id", "version", "sha256"],
        },
        "fingerprint": _fingerprint("model", "production-local-model-cache-v1", "1.0.0"),
        "license_policy": {
            "default_license": "operator_supplied",
            "requires_operator_attestation": True,
            "redistribution_allowed": "operator_defined",
        },
        "resource_policy": {
            "storage": "local_cache",
            "network_required": False,
            "gpu_required": False,
            "live_model_calls": False,
            "package_installation": False,
            "max_size_mb": 2048,
        },
        "reproducibility_notes": [
            "Cached model artifacts must be verified by sha256 before use.",
            "Bundles should record the selected cache fingerprint and runtime configuration.",
        ],
        "live_model_calls": False,
        "network_required": False,
        "gpu_required": False,
        "intended_use": "Reuse operator-approved local model artifacts without downloads or hosted inference.",
    },
    {
        "id": "optional-external-model-adapter-template-v1",
        "name": "Optional external model adapter template",
        "status": "disabled_template",
        "profile": PRODUCTION_PROFILE,
        "version": "0.1.0",
        "versioning": {
            "scheme": "adapter_contract",
            "stability": "disabled until explicitly implemented",
            "change_policy": "Enabling requires a new profile, policy review, fingerprint, and license metadata.",
        },
        "implementation": "external_adapter_placeholder",
        "adapter": {
            "type": "optional_external",
            "enabled": False,
            "network_required": False,
            "provider": "placeholder",
            "activation_policy": "not_enabled_for_production_local",
        },
        "fingerprint": _fingerprint("model", "optional-external-model-adapter-template-v1", "0.1.0"),
        "license_policy": {
            "default_license": "unknown_until_operator_attested",
            "requires_operator_attestation": True,
            "redistribution_allowed": False,
        },
        "resource_policy": {
            "storage": "none_until_enabled",
            "network_required": False,
            "gpu_required": False,
            "live_model_calls": False,
            "package_installation": False,
        },
        "reproducibility_notes": [
            "This is a registry contract placeholder only; production-local audit verifies it is disabled.",
            "No hosted inference or model download is performed by AIRA under the production-local profile.",
        ],
        "live_model_calls": False,
        "network_required": False,
        "gpu_required": False,
        "intended_use": "Document the external model adapter contract without enabling live or external models.",
    },
]


BENCHMARKS: list[dict[str, Any]] = [
    {
        "id": "fixture-classification-smoke",
        "name": "AIRA deterministic fixture benchmark",
        "status": "mvp",
        "dataset_id": "fixture-ai-classification",
        "model_ids": ["fixture-threshold-classifier", "fixture-majority-baseline"],
        "metric_ids": ["accuracy", "accuracy_delta"],
        "network_required": False,
        "emits_bundle_type": "aira_result_bundle",
    },
    {
        "id": "local-text-outcome-classification",
        "name": "AIRA deterministic local text outcome benchmark",
        "status": "local_deterministic",
        "dataset_id": "local-experiment-outcomes-v1",
        "model_ids": [
            "deterministic-keyword-outcome-classifier-v1",
            "deterministic-pass-prior-baseline-v1",
            "deterministic-keyword-no-negative-ablation-v1",
        ],
        "metric_ids": ["accuracy", "macro_f1", "baseline_accuracy", "accuracy_delta", "ablation_error_count"],
        "network_required": False,
        "external_datasets_required": False,
        "gpu_required": False,
        "live_model_calls": False,
        "emits_bundle_type": "aira_result_bundle",
        "emits_artifact_kinds": [
            "benchmark_report",
            "ablation_report",
            "error_analysis",
            "provenance",
            "run_ledger",
            "experiment_memory",
        ],
        "entrypoint": "python3 -m aira run-local-benchmark",
    },
]


PRODUCTION_BENCHMARKS: list[dict[str, Any]] = [
    {
        "id": "production-local-plan-execution",
        "name": "AIRA production-local plan execution benchmark",
        "status": "production_local",
        "profile": PRODUCTION_PROFILE,
        "version": "1.0.0",
        "versioning": {
            "scheme": "semver",
            "stability": "profile pinned",
            "change_policy": "Plan schema, profile policy, or emitted artifact contract changes require a new fingerprint.",
        },
        "dataset_id": "operator-supplied-production-plan",
        "model_ids": [
            "production-local-controlled-python-runner-v1",
            "operator-supplied-model-artifact-v1",
            "production-local-model-cache-v1",
        ],
        "metric_ids": ["task_count", "passed_task_count", "failed_task_count", "skipped_task_count"],
        "adapter": {
            "type": "production_local_runner",
            "enabled": True,
            "network_required": False,
            "entrypoint": "python3 -m aira experiments run --profile production-local",
        },
        "fingerprint": _fingerprint("benchmark", "production-local-plan-execution", "1.0.0"),
        "license_policy": {
            "default_license": "operator_supplied",
            "requires_operator_attestation": True,
            "redistribution_allowed": "operator_defined",
        },
        "resource_policy": {
            "network_required": False,
            "external_datasets_required": False,
            "gpu_required": False,
            "live_model_calls": False,
            "package_installation": False,
            "max_task_timeout_seconds": 30,
            "max_cpu_threads": 2,
        },
        "reproducibility_notes": [
            "The benchmark is reproducible from the production plan, profile policy, materialized task artifacts, and fingerprints.",
            "External dataset and model adapters remain disabled under production-local.",
        ],
        "network_required": False,
        "external_datasets_required": False,
        "gpu_required": False,
        "live_model_calls": False,
        "emits_bundle_type": "aira_result_bundle",
        "emits_artifact_kinds": [
            "production_plan",
            "policy_report",
            "execution_trace",
            "task_summary",
            "provenance",
            "reproduction_status",
            "run_ledger",
        ],
        "entrypoint": "python3 -m aira experiments run --profile production-local",
    }
]


PRODUCTION_OPEN_DATASETS: list[dict[str, Any]] = [
    {
        "id": "external-download-dataset-adapter-v1",
        "name": "External dataset download adapter",
        "status": "production_open",
        "source": "external_download",
        "profile": PRODUCTION_OPEN_PROFILE,
        "version": "1.0.0",
        "versioning": {
            "scheme": "content_addressed_when_available",
            "stability": "operator controlled",
            "change_policy": "Downloaded datasets must record source URL, license, retrieval time, and fingerprints when available.",
        },
        "adapter": {
            "type": "optional_external",
            "enabled": True,
            "network_required": True,
            "activation_policy": "enabled_by_production_open_profile",
        },
        "fingerprint": _fingerprint("dataset", "external-download-dataset-adapter-v1", "1.0.0"),
        "license_policy": {
            "default_license": "source_declared",
            "requires_operator_attestation": True,
            "redistribution_allowed": "source_license_defined",
        },
        "resource_policy": {
            "storage": "download_or_cache",
            "network_required": True,
            "external_datasets_required": True,
            "gpu_required": False,
            "max_size_mb": "operator_declared",
        },
        "reproducibility_notes": [
            "Production-open permits remote dataset downloads and records retrieval metadata in the result bundle.",
            "Operators must preserve source URL, license, and fingerprints for publishable claims.",
        ],
        "network_required": True,
        "external_datasets_required": True,
        "intended_use": "Download and use external datasets for unrestricted AIRA experiments.",
    }
]


PRODUCTION_OPEN_MODELS: list[dict[str, Any]] = [
    {
        "id": "production-open-python-runner-v1",
        "name": "Production-open Python and command runner",
        "status": "production_open",
        "profile": PRODUCTION_OPEN_PROFILE,
        "version": "1.0.0",
        "versioning": {
            "scheme": "semver",
            "stability": "profile pinned",
            "change_policy": "Execution surface changes require a new runner fingerprint.",
        },
        "implementation": "aira.production_runner.run_production_experiment",
        "adapter": {
            "type": "builtin_runner",
            "enabled": True,
            "network_required": True,
            "live_model_calls": True,
        },
        "fingerprint": _fingerprint("model", "production-open-python-runner-v1", "1.0.0"),
        "license_policy": {
            "default_license": "project_license",
            "requires_operator_attestation": False,
            "redistribution_allowed": True,
        },
        "resource_policy": {
            "network_required": True,
            "gpu_required": True,
            "live_model_calls": True,
            "package_installation": True,
            "max_cpu_threads": "host_available",
            "max_task_timeout_seconds": 3600,
        },
        "reproducibility_notes": [
            "The production-open runner can install packages, run command tasks, download data or models, use GPU, and call live model APIs.",
            "Runs are not assumed deterministic; bundles must capture versions, fingerprints, and external service metadata.",
        ],
        "live_model_calls": True,
        "network_required": True,
        "gpu_required": True,
        "intended_use": "Execute unrestricted AIRA AI/ML experiment plans.",
    },
    {
        "id": "hosted-model-api-adapter-v1",
        "name": "Hosted model API adapter",
        "status": "production_open",
        "profile": PRODUCTION_OPEN_PROFILE,
        "version": "1.0.0",
        "implementation": "operator_configured_hosted_api",
        "adapter": {
            "type": "hosted_model_api",
            "enabled": True,
            "network_required": True,
            "activation_policy": "operator_supplied_credentials",
        },
        "fingerprint": _fingerprint("model", "hosted-model-api-adapter-v1", "1.0.0"),
        "license_policy": {
            "default_license": "provider_terms",
            "requires_operator_attestation": True,
            "redistribution_allowed": "provider_terms_defined",
        },
        "resource_policy": {
            "network_required": True,
            "gpu_required": False,
            "live_model_calls": True,
            "package_installation": True,
        },
        "reproducibility_notes": [
            "Hosted model calls must record provider, model id, version/date, request parameters, and response fingerprints where possible.",
        ],
        "live_model_calls": True,
        "network_required": True,
        "gpu_required": False,
        "intended_use": "Call externally hosted models during production-open experiments.",
    },
]


PRODUCTION_OPEN_BENCHMARKS: list[dict[str, Any]] = [
    {
        "id": "production-open-plan-execution",
        "name": "AIRA production-open plan execution benchmark",
        "status": "production_open",
        "profile": PRODUCTION_OPEN_PROFILE,
        "version": "1.0.0",
        "versioning": {
            "scheme": "semver",
            "stability": "profile pinned",
            "change_policy": "Plan schema, open-profile policy, or emitted artifact contract changes require a new fingerprint.",
        },
        "dataset_id": "external-download-dataset-adapter-v1",
        "model_ids": [
            "production-open-python-runner-v1",
            "hosted-model-api-adapter-v1",
        ],
        "metric_ids": ["task_count", "passed_task_count", "failed_task_count", "skipped_task_count"],
        "adapter": {
            "type": "production_open_runner",
            "enabled": True,
            "network_required": True,
            "entrypoint": "python3 -m aira experiments run --profile production-open",
        },
        "fingerprint": _fingerprint("benchmark", "production-open-plan-execution", "1.0.0"),
        "license_policy": {
            "default_license": "operator_supplied",
            "requires_operator_attestation": True,
            "redistribution_allowed": "operator_defined",
        },
        "resource_policy": {
            "network_required": True,
            "external_datasets_required": True,
            "gpu_required": True,
            "live_model_calls": True,
            "package_installation": True,
            "max_task_timeout_seconds": 3600,
            "max_cpu_threads": "host_available",
        },
        "reproducibility_notes": [
            "Production-open intentionally restores the broad legacy ARA experiment surface inside AIRA.",
            "Bundles must record external source, package, model/API, and hardware metadata for publishable claims.",
        ],
        "network_required": True,
        "external_datasets_required": True,
        "gpu_required": True,
        "live_model_calls": True,
        "emits_bundle_type": "aira_result_bundle",
        "emits_artifact_kinds": [
            "production_plan",
            "policy_report",
            "execution_trace",
            "task_summary",
            "provenance",
            "reproduction_status",
            "run_ledger",
        ],
        "entrypoint": "python3 -m aira experiments run --profile production-open",
    }
]


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def registry_payload(profile: str | None = None) -> dict[str, Any]:
    """Return registry entries as JSON-serializable data.

    The default payload intentionally preserves the fixture/local registry view.
    Production entries are included only through explicit production profiles to
    keep deterministic smoke behavior stable while allowing production-open runs
    to opt into external resources.
    """
    requested_profile = profile or "default"
    datasets = deepcopy(DATASETS)
    models = deepcopy(MODELS)
    benchmarks = deepcopy(BENCHMARKS)
    if requested_profile != "default":
        if requested_profile not in PRODUCTION_PROFILES:
            raise ValueError(f"Unsupported registry profile: {profile}.")
        if requested_profile == PRODUCTION_PROFILE:
            datasets.extend(deepcopy(PRODUCTION_DATASETS))
            models.extend(deepcopy(PRODUCTION_MODELS))
            benchmarks.extend(deepcopy(PRODUCTION_BENCHMARKS))
        else:
            datasets.extend(deepcopy(PRODUCTION_OPEN_DATASETS))
            models.extend(deepcopy(PRODUCTION_OPEN_MODELS))
            benchmarks.extend(deepcopy(PRODUCTION_OPEN_BENCHMARKS))
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "profile": requested_profile,
        "datasets": datasets,
        "models": models,
        "benchmarks": benchmarks,
    }


def production_registry_payload(profile: str = PRODUCTION_PROFILE) -> dict[str, Any]:
    return registry_payload(profile)


def _entry_id(entry: dict[str, Any]) -> str:
    return str(entry.get("id", "")).strip()


def _check_required_metadata(kind: str, entry: dict[str, Any], errors: list[str], checks: list[dict[str, Any]]) -> None:
    entry_id = _entry_id(entry)
    required = ("version", "fingerprint", "license_policy", "resource_policy", "reproducibility_notes")
    missing = [field for field in required if field not in entry]
    if missing:
        errors.append(f"{kind} {entry_id} is missing required metadata fields: {missing}.")
        checks.append({"id": f"{kind}:{entry_id}:metadata", "status": "fail", "missing": missing})
        return
    fingerprint = entry.get("fingerprint")
    notes = entry.get("reproducibility_notes")
    if not isinstance(fingerprint, dict) or fingerprint.get("algorithm") != "sha256" or not fingerprint.get("value"):
        errors.append(f"{kind} {entry_id} has an invalid sha256 fingerprint.")
        checks.append({"id": f"{kind}:{entry_id}:fingerprint", "status": "fail"})
        return
    if not isinstance(notes, list) or not notes:
        errors.append(f"{kind} {entry_id} must include reproducibility notes.")
        checks.append({"id": f"{kind}:{entry_id}:reproducibility", "status": "fail"})
        return
    checks.append({"id": f"{kind}:{entry_id}:metadata", "status": "pass"})


def _check_production_policy(kind: str, entry: dict[str, Any], errors: list[str], checks: list[dict[str, Any]]) -> None:
    entry_id = _entry_id(entry)
    if entry.get("profile") == PRODUCTION_OPEN_PROFILE:
        resource_policy = entry.get("resource_policy", {})
        if not isinstance(resource_policy, dict):
            errors.append(f"{kind} {entry_id} is missing a production-open resource policy.")
            checks.append({"id": f"{kind}:{entry_id}:policy", "status": "fail"})
            return
        checks.append({"id": f"{kind}:{entry_id}:policy", "status": "pass", "open_profile": True})
        return
    resource_policy = entry.get("resource_policy", {})
    blocked_flags = {
        "network_required": entry.get("network_required"),
        "external_datasets_required": entry.get("external_datasets_required"),
        "gpu_required": entry.get("gpu_required"),
        "live_model_calls": entry.get("live_model_calls"),
    }
    if isinstance(resource_policy, dict):
        for key in blocked_flags:
            blocked_flags[key] = blocked_flags[key] or resource_policy.get(key)
    adapter = entry.get("adapter", {})
    enabled_external = (
        isinstance(adapter, dict)
        and adapter.get("type") == "optional_external"
        and adapter.get("enabled") is True
    )
    failures = [flag for flag, value in blocked_flags.items() if value is True]
    if enabled_external:
        failures.append("optional_external_enabled")
    if failures:
        errors.append(f"{kind} {entry_id} violates production-local resource policy: {failures}.")
        checks.append({"id": f"{kind}:{entry_id}:policy", "status": "fail", "failures": failures})
        return
    checks.append({"id": f"{kind}:{entry_id}:policy", "status": "pass"})


def audit_registry(profile: str) -> dict[str, Any]:
    """Audit a registry profile for production reproducibility metadata."""
    if profile not in PRODUCTION_PROFILES:
        return {
            "schema_version": PRODUCTION_REGISTRY_SCHEMA_VERSION,
            "profile": profile,
            "status": "failed",
            "valid": False,
            "errors": [f"Unsupported registry audit profile: {profile}."],
            "warnings": [],
            "checks": [],
        }

    payload = production_registry_payload(profile)
    errors: list[str] = []
    warnings: list[str] = []
    checks: list[dict[str, Any]] = []
    datasets = {_entry_id(entry): entry for entry in payload["datasets"]}
    models = {_entry_id(entry): entry for entry in payload["models"]}
    production_entries = {
        "dataset": [entry for entry in payload["datasets"] if entry.get("profile") == profile],
        "model": [entry for entry in payload["models"] if entry.get("profile") == profile],
        "benchmark": [entry for entry in payload["benchmarks"] if entry.get("profile") == profile],
    }

    for kind, entries in production_entries.items():
        if not entries:
            errors.append(f"No {profile} {kind} entries are registered.")
            checks.append({"id": f"{kind}:presence", "status": "fail"})
            continue
        checks.append({"id": f"{kind}:presence", "status": "pass", "count": len(entries)})
        for entry in entries:
            _check_required_metadata(kind, entry, errors, checks)
            _check_production_policy(kind, entry, errors, checks)

    adapter_types = {
        str(entry.get("adapter", {}).get("type"))
        for entries in production_entries.values()
        for entry in entries
        if isinstance(entry.get("adapter"), dict)
    }
    required_adapters = (
        ("local_cache", "operator_supplied_artifact", "optional_external")
        if profile == PRODUCTION_PROFILE
        else ("optional_external", "builtin_runner", "hosted_model_api", "production_open_runner")
    )
    for required_adapter in required_adapters:
        if required_adapter not in adapter_types:
            errors.append(f"Production registry is missing adapter type: {required_adapter}.")
            checks.append({"id": f"adapter:{required_adapter}", "status": "fail"})
        else:
            checks.append({"id": f"adapter:{required_adapter}", "status": "pass"})

    for benchmark in production_entries["benchmark"]:
        benchmark_id = _entry_id(benchmark)
        dataset_id = str(benchmark.get("dataset_id", ""))
        missing_models = [model_id for model_id in benchmark.get("model_ids", []) if model_id not in models]
        if dataset_id not in datasets or missing_models:
            errors.append(
                f"Benchmark {benchmark_id} references missing dataset/model entries: "
                f"dataset={dataset_id!r}, models={missing_models!r}."
            )
            checks.append({"id": f"benchmark:{benchmark_id}:references", "status": "fail"})
        else:
            checks.append({"id": f"benchmark:{benchmark_id}:references", "status": "pass"})

    snapshot = {
        "datasets": production_entries["dataset"],
        "models": production_entries["model"],
        "benchmarks": production_entries["benchmark"],
    }
    valid = not errors
    return {
        "schema_version": PRODUCTION_REGISTRY_SCHEMA_VERSION,
        "profile": profile,
        "status": "passed" if valid else "failed",
        "valid": valid,
        "registry_sha256": _canonical_digest(snapshot),
        "counts": {
            "datasets": len(payload["datasets"]),
            "models": len(payload["models"]),
            "benchmarks": len(payload["benchmarks"]),
            "production_datasets": len(production_entries["dataset"]),
            "production_models": len(production_entries["model"]),
            "production_benchmarks": len(production_entries["benchmark"]),
        },
        "adapter_types": sorted(adapter_types),
        "production_registry": snapshot,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }
