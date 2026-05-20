"""AIRA result bundle writer and validator."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any


BUNDLE_SCHEMA_VERSION = "ara.result_bundle.v1"
VALIDATION_SCHEMA_VERSION = "aira.bundle_validation.v1"
ARA_HANDOFF_SCHEMA_VERSION = "aira.ara_handoff.v1"
ARA_GATE_PROFILE = "ara-public-bundle-reproduction-gate.v1"
DEFAULT_VALIDATION_PROFILE = "aira-mvp"
ARA_PRODUCTION_VALIDATION_PROFILE = "ara-production"
ARA_PRODUCTION_OPEN_VALIDATION_PROFILE = "ara-production-open"
VALIDATION_PROFILES = {DEFAULT_VALIDATION_PROFILE, ARA_PRODUCTION_VALIDATION_PROFILE, ARA_PRODUCTION_OPEN_VALIDATION_PROFILE}
REQUIRED_FILES = [
    "bundle_manifest.json",
    "artifact_manifest.json",
    "writing_brief.md",
    "limitations.md",
    "claims.json",
]
REPRODUCED_STATUSES = {"reproduced", "confirmed", "passed", "pass"}
CONFIRMED_STATUSES = {"confirmed", "verified", "reproduced", "passed", "pass", "supported"}


@dataclass(frozen=True)
class BundleValidationResult:
    path: Path
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: list[dict[str, str]] = field(default_factory=list)
    files: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    bundle_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "path": str(self.path),
            "valid": self.valid,
            "bundle_type": self.bundle_type,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "checks": list(self.checks),
            "files": self.files,
            "metadata": self.metadata,
        }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path, errors: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"Could not read JSON file {path.name}: {exc}")
        return None


def _check(checks: list[dict[str, str]], check_id: str, status: str, message: str) -> None:
    checks.append({"id": check_id, "status": status, "message": message})


def _file_report(bundle_path: Path, relative: str) -> dict[str, Any]:
    file_path = bundle_path / relative
    report: dict[str, Any] = {
        "required": relative in REQUIRED_FILES,
        "present": file_path.exists(),
        "is_file": file_path.is_file() if file_path.exists() else False,
    }
    if file_path.exists() and file_path.is_file():
        report["size_bytes"] = file_path.stat().st_size
    return report


def _safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value.strip()) and not path.is_absolute() and ".." not in path.parts


def _required_string(mapping: dict[str, Any], key: str, source: str, errors: list[str]) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{source} field `{key}` is required and must be a non-empty string.")
        return ""
    return value.strip()


def _validate_artifact_manifest(
    payload: Any,
    *,
    bundle_path: Path,
    errors: list[str],
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    if not isinstance(payload, dict):
        errors.append("artifact_manifest.json must contain a JSON object.")
        return set(), {}
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        errors.append("artifact_manifest.json field `artifacts` must be a list.")
        return set(), {}

    artifact_ids: set[str] = set()
    artifact_details: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(artifacts):
        prefix = f"artifact_manifest.json artifacts[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be an object.")
            continue
        artifact_id = item.get("artifact_id", item.get("id"))
        path_value = item.get("path")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            errors.append(f"{prefix}.artifact_id is required and must be a non-empty string.")
            continue
        if artifact_id in artifact_ids:
            errors.append(f"{prefix}.artifact_id duplicates an earlier artifact id: {artifact_id}.")
        artifact_ids.add(artifact_id)
        if not isinstance(path_value, str) or not path_value.strip():
            errors.append(f"{prefix}.path is required and must be a non-empty string.")
            path_value = ""
        elif not _safe_relative_path(path_value):
            errors.append(f"{prefix}.path must be a safe relative path within the bundle.")
        else:
            artifact_path = bundle_path / path_value
            if not artifact_path.exists():
                errors.append(f"{prefix}.path does not exist in bundle: {path_value}")
            elif not artifact_path.is_file():
                errors.append(f"{prefix}.path must point to a file: {path_value}")
        if "kind" in item and not isinstance(item["kind"], str):
            errors.append(f"{prefix}.kind must be a string when present.")
        if "description" in item and not isinstance(item["description"], str):
            errors.append(f"{prefix}.description must be a string when present.")
        detail = {
            "artifact_id": artifact_id,
            "path": path_value,
            "kind": item.get("kind") if isinstance(item.get("kind"), str) else "",
            "description": item.get("description") if isinstance(item.get("description"), str) else "",
        }
        artifact_details[artifact_id] = detail
        if path_value:
            artifact_details[path_value] = detail
    return artifact_ids, artifact_details


def _artifact_tokens(refs: list[str], artifact_details: dict[str, dict[str, Any]]) -> list[str]:
    tokens: list[str] = []
    for ref in refs:
        tokens.append(ref)
        detail = artifact_details.get(ref, {})
        for key in ("artifact_id", "path", "kind", "description"):
            value = detail.get(key)
            if isinstance(value, str):
                tokens.append(value)
    return [token.lower() for token in tokens]


def _has_reproduction_artifact(refs: list[str], artifact_details: dict[str, dict[str, Any]]) -> bool:
    return any(
        "reproduction_status" in token or "reproduced" in token
        for token in _artifact_tokens(refs, artifact_details)
    )


def _validate_claims(
    payload: Any,
    *,
    artifact_ids: set[str],
    artifact_details: dict[str, dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> int:
    if not isinstance(payload, dict):
        errors.append("claims.json must contain a JSON object.")
        return 0
    claims = payload.get("claims")
    if not isinstance(claims, list):
        errors.append("claims.json must contain a `claims` list.")
        return 0
    if not claims:
        warnings.append("claims.json contains no claims.")
    for index, claim in enumerate(claims):
        prefix = f"claims[{index}]"
        if not isinstance(claim, dict):
            errors.append(f"{prefix} must be an object.")
            continue
        claim_id = claim.get("claim_id", claim.get("id"))
        text = claim.get("claim", claim.get("text"))
        status = claim.get("status")
        supported_by = claim.get("supported_by", claim.get("artifacts"))
        reproduction_status = claim.get("reproduction_status")
        if not isinstance(claim_id, str) or not claim_id.strip():
            errors.append(f"{prefix}.claim_id is required and must be a non-empty string.")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{prefix}.claim is required and must be a non-empty string.")
        if not isinstance(status, str) or not status.strip():
            errors.append(f"{prefix}.status is required and must be a non-empty string.")
            status = ""
        else:
            status = status.strip()
        if supported_by is None:
            errors.append(f"{prefix}.supported_by is required.")
            refs: list[str] = []
        elif not isinstance(supported_by, list):
            errors.append(f"{prefix}.supported_by must be a list.")
            refs = []
        else:
            refs = []
            for ref_index, ref in enumerate(supported_by):
                if not isinstance(ref, str) or not ref.strip():
                    errors.append(f"{prefix}.supported_by[{ref_index}] must be a non-empty string.")
                else:
                    refs.append(ref.strip())
        missing_refs = sorted(ref for ref in refs if ref not in artifact_ids and ref not in artifact_details)
        if missing_refs:
            errors.append(f"{prefix}.supported_by references undeclared artifacts: {missing_refs}.")
        limitations = claim.get("limitations")
        if not isinstance(limitations, list):
            errors.append(f"{prefix}.limitations is required and must be a list.")
        elif not limitations:
            warnings.append(f"{prefix}.limitations is empty.")
        elif any(not isinstance(item, str) or not item.strip() for item in limitations):
            errors.append(f"{prefix}.limitations must contain only non-empty strings.")

        if status in CONFIRMED_STATUSES:
            if not isinstance(reproduction_status, str) or reproduction_status.strip() not in REPRODUCED_STATUSES:
                errors.append(f"{prefix} marks an AIRA claim confirmed without a reproduced reproduction_status.")
            elif not _has_reproduction_artifact(refs, artifact_details):
                errors.append(f"{prefix} marks an AIRA claim confirmed without a reproduction status artifact.")
    return len(claims)


def _artifact_paths_for(
    artifact_details: dict[str, dict[str, Any]],
    *,
    artifact_ids: set[str],
    kinds: set[str],
) -> list[str]:
    paths = {
        detail["path"]
        for detail in artifact_details.values()
        if isinstance(detail.get("path"), str)
        and detail["path"]
        and (detail.get("artifact_id") in artifact_ids or detail.get("kind") in kinds)
    }
    return sorted(paths)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _validate_false_flag(
    mapping: dict[str, Any],
    key: str,
    source: str,
    errors: list[str],
) -> None:
    if mapping.get(key) is not False:
        errors.append(f"{source} field `{key}` must be false for deterministic local AIRA bundles.")


def _validate_provenance_artifact(payload: Any, source: str, errors: list[str]) -> None:
    if not isinstance(payload, dict):
        errors.append(f"{source} must contain a JSON object.")
        return
    for key in ("schema_version", "run_id", "benchmark_id", "dataset_id", "model_id"):
        _required_string(payload, key, source, errors)
    fingerprints = payload.get("input_fingerprints")
    if not isinstance(fingerprints, dict):
        errors.append(f"{source} field `input_fingerprints` must be an object.")
    else:
        for key in ("dataset_sha256", "model_config_sha256", "registry_snapshot_sha256"):
            if not _is_sha256(fingerprints.get(key)):
                errors.append(f"{source} input_fingerprints.{key} must be a lowercase sha256 hex digest.")
    execution = payload.get("execution")
    if not isinstance(execution, dict):
        errors.append(f"{source} field `execution` must be an object.")
    else:
        for key in ("runner", "command", "package_version", "python_version"):
            _required_string(execution, key, f"{source}.execution", errors)
    determinism = payload.get("determinism")
    if not isinstance(determinism, dict):
        errors.append(f"{source} field `determinism` must be an object.")
    else:
        dynamic_flags = {
            key: determinism.get(key) is True
            for key in ("network_required", "external_datasets_required", "gpu_required", "live_model_calls")
        }
        if determinism.get("deterministic") is not True and not any(dynamic_flags.values()):
            errors.append(f"{source} determinism.deterministic must be true unless dynamic open-profile resources are declared.")
        if determinism.get("deterministic") is True:
            for key in dynamic_flags:
                _validate_false_flag(determinism, key, f"{source}.determinism", errors)


def _validate_run_ledger_entry(payload: Any, source: str, errors: list[str]) -> None:
    if not isinstance(payload, dict):
        errors.append(f"{source} must contain a JSON object.")
        return
    for key in ("schema_version", "run_id", "status", "bundle_type", "benchmark_id", "dataset_id", "model_id"):
        _required_string(payload, key, source, errors)
    if payload.get("bundle_type") != "aira_result_bundle":
        errors.append(f"{source} field `bundle_type` must be `aira_result_bundle`.")
    if payload.get("status") not in {"passed", "failed"}:
        errors.append(f"{source} field `status` must be `passed` or `failed`.")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        errors.append(f"{source} field `metrics` must be a non-empty object.")
    reproducibility = payload.get("reproducibility")
    if not isinstance(reproducibility, dict):
        errors.append(f"{source} field `reproducibility` must be an object.")
    else:
        dynamic_flags = {
            key: reproducibility.get(key) is True
            for key in ("network_required", "external_datasets_required", "gpu_required", "live_model_calls")
        }
        if reproducibility.get("deterministic") is not True and not any(dynamic_flags.values()):
            errors.append(f"{source} reproducibility.deterministic must be true unless dynamic open-profile resources are declared.")
        if reproducibility.get("deterministic") is True:
            for key in dynamic_flags:
                _validate_false_flag(reproducibility, key, f"{source}.reproducibility", errors)


def _run_id_from_ledger_payload(payload: Any) -> str | None:
    if isinstance(payload, dict) and isinstance(payload.get("run_id"), str) and payload["run_id"].strip():
        return payload["run_id"].strip()
    return None


def _validate_run_ledger_artifact(path: Path, source: str, errors: list[str]) -> set[str]:
    run_ids: set[str] = set()
    if path.suffix == ".jsonl":
        try:
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except Exception as exc:
            errors.append(f"Could not read JSONL file {source}: {exc}")
            return run_ids
        if not lines:
            errors.append(f"{source} must contain at least one JSONL entry.")
            return run_ids
        for index, line in enumerate(lines):
            try:
                payload = json.loads(line)
            except Exception as exc:
                errors.append(f"{source} line {index + 1} is not valid JSON: {exc}")
                continue
            _validate_run_ledger_entry(payload, f"{source} line {index + 1}", errors)
            run_id = _run_id_from_ledger_payload(payload)
            if run_id:
                run_ids.add(run_id)
        return run_ids
    payload = _read_json(path, errors)
    _validate_run_ledger_entry(payload, source, errors)
    run_id = _run_id_from_ledger_payload(payload)
    if run_id:
        run_ids.add(run_id)
    return run_ids


def _gate_input_path_refs(payload: Any, errors: list[str]) -> dict[str, str]:
    if not isinstance(payload, dict):
        errors.append("ara_handoff.required_gate_inputs must be an object.")
        return {}
    refs: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(value, str) or not value.strip():
            errors.append(f"ara_handoff.required_gate_inputs.{key} must be a non-empty string path.")
            continue
        if not _safe_relative_path(value):
            errors.append(f"ara_handoff.required_gate_inputs.{key} must be a safe relative bundle path.")
            continue
        refs[key] = value.strip()
    return refs


def _validate_ara_handoff_artifact(
    payload: Any,
    *,
    bundle_path: Path,
    artifact_details: dict[str, dict[str, Any]],
    errors: list[str],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "profile": None,
        "required_inputs": {},
        "required_inputs_present": False,
    }
    if not isinstance(payload, dict):
        errors.append("ara_handoff artifact must contain a JSON object.")
        return metadata

    for key in ("schema_version", "consumer", "gate_profile", "bundle_schema_version", "bundle_type", "run_id"):
        _required_string(payload, key, "ara_handoff", errors)
    if payload.get("schema_version") != ARA_HANDOFF_SCHEMA_VERSION:
        errors.append(f"ara_handoff.schema_version must be `{ARA_HANDOFF_SCHEMA_VERSION}`.")
    if payload.get("consumer") != "ara":
        errors.append("ara_handoff.consumer must be `ara`.")
    if payload.get("gate_profile") != ARA_GATE_PROFILE:
        errors.append(f"ara_handoff.gate_profile must be `{ARA_GATE_PROFILE}`.")
    if payload.get("bundle_schema_version") != BUNDLE_SCHEMA_VERSION:
        errors.append(f"ara_handoff.bundle_schema_version must be `{BUNDLE_SCHEMA_VERSION}`.")
    if payload.get("bundle_type") != "aira_result_bundle":
        errors.append("ara_handoff.bundle_type must be `aira_result_bundle`.")
    if payload.get("status") != "ready":
        errors.append("ara_handoff.status must be `ready`.")

    reproducibility = payload.get("reproducibility")
    if not isinstance(reproducibility, dict):
        errors.append("ara_handoff.reproducibility must be an object.")
    else:
        if reproducibility.get("deterministic") is not True:
            errors.append("ara_handoff.reproducibility.deterministic must be true.")
        for key in ("network_required", "external_datasets_required", "gpu_required", "live_model_calls"):
            _validate_false_flag(reproducibility, key, "ara_handoff.reproducibility", errors)
        fingerprints = reproducibility.get("input_fingerprints")
        if not isinstance(fingerprints, dict):
            errors.append("ara_handoff.reproducibility.input_fingerprints must be an object.")
        else:
            for key in ("dataset_sha256", "model_config_sha256", "registry_snapshot_sha256"):
                if not _is_sha256(fingerprints.get(key)):
                    errors.append(f"ara_handoff.reproducibility.input_fingerprints.{key} must be a sha256 hex digest.")

    claim_gate = payload.get("claim_gate")
    if not isinstance(claim_gate, dict):
        errors.append("ara_handoff.claim_gate must be an object.")
    else:
        for key in ("confirmed_claims_require_reproduced_status", "confirmed_claims_require_reproduction_artifact"):
            if claim_gate.get(key) is not True:
                errors.append(f"ara_handoff.claim_gate.{key} must be true.")

    required_inputs = _gate_input_path_refs(payload.get("required_gate_inputs"), errors)
    expected_inputs = {
        "bundle_manifest",
        "artifact_manifest",
        "claims",
        "writing_brief",
        "limitations",
        "reproducibility_notes",
        "reproduction_status",
        "provenance",
        "run_ledger_entry",
        "run_ledger",
    }
    missing_keys = sorted(expected_inputs - set(required_inputs))
    if missing_keys:
        errors.append(f"ara_handoff.required_gate_inputs is missing keys: {missing_keys}.")
    declared_paths = {detail["path"] for detail in artifact_details.values() if isinstance(detail.get("path"), str)}
    allowed_paths = set(REQUIRED_FILES) | declared_paths
    missing_paths = sorted(value for value in required_inputs.values() if value not in allowed_paths)
    if missing_paths:
        errors.append(f"ara_handoff.required_gate_inputs references undeclared bundle paths: {missing_paths}.")
    missing_file_keys: list[str] = []
    for key, relative in required_inputs.items():
        candidate = bundle_path / relative
        if not candidate.is_file():
            missing_file_keys.append(key)
            errors.append(f"ara_handoff.required_gate_inputs.{key} does not point to a bundle file: {relative}")

    metadata["profile"] = payload.get("gate_profile")
    metadata["required_inputs"] = required_inputs
    metadata["missing_required_input_keys"] = missing_keys
    metadata["missing_required_input_files"] = sorted(missing_file_keys)
    metadata["required_inputs_present"] = not missing_keys and not missing_paths and not missing_file_keys
    return metadata


def _validate_ara_production_profile(
    *,
    bundle_path: Path,
    metadata: dict[str, Any],
    artifact_ids: set[str],
    artifact_details: dict[str, dict[str, Any]],
    errors: list[str],
    checks: list[dict[str, str]],
) -> None:
    before = len(errors)
    manifest = metadata.get("bundle_manifest")
    if not isinstance(manifest, dict):
        errors.append("ara-production profile requires a valid bundle_manifest.json object.")
        manifest = {}

    production_runner = manifest.get("production_runner")
    if not isinstance(production_runner, dict):
        errors.append("ara-production profile requires bundle_manifest.json.production_runner.")
    elif production_runner.get("profile") != "production-local":
        errors.append("ara-production profile requires production_runner.profile to be `production-local`.")

    production_evaluation = manifest.get("production_evaluation")
    if not isinstance(production_evaluation, dict):
        errors.append("ara-production profile requires bundle_manifest.json.production_evaluation.")
    elif production_evaluation.get("status") != "passed":
        errors.append("ara-production profile requires production_evaluation.status to be `passed`.")

    ara_handoff = manifest.get("ara_handoff")
    if not isinstance(ara_handoff, dict):
        errors.append("ara-production profile requires bundle_manifest.json.ara_handoff.")
    else:
        if ara_handoff.get("validation_profile") != ARA_PRODUCTION_VALIDATION_PROFILE:
            errors.append("bundle_manifest.json ara_handoff.validation_profile must be `ara-production`.")
        if ara_handoff.get("validation_command") != (
            "python3 -m aira bundles validate <bundle> --profile ara-production --json"
        ):
            errors.append("bundle_manifest.json ara_handoff.validation_command must use the ara-production profile.")

    ara_gate = metadata.get("ara_gate")
    if not isinstance(ara_gate, dict) or not ara_gate.get("required_inputs_present"):
        errors.append("ara-production profile requires complete ARA handoff required gate inputs.")
        required_inputs: dict[str, str] = {}
    else:
        required_inputs = ara_gate.get("required_inputs") if isinstance(ara_gate.get("required_inputs"), dict) else {}

    production_required_inputs = {
        "production_plan",
        "policy_report",
        "execution_trace",
        "task_summary",
        "production_evaluation_metrics",
        "production_ablation_matrix",
        "production_error_taxonomy",
        "production_statistical_tests",
        "production_report_summary",
        "memory_index",
        "memory_runs",
        "memory_failures",
        "memory_fingerprints",
        "memory_outcomes",
        "memory_reflections",
    }
    missing_inputs = sorted(production_required_inputs - set(required_inputs))
    if missing_inputs:
        errors.append(f"ara-production required_gate_inputs is missing keys: {missing_inputs}.")

    required_artifact_ids = {
        "ara_handoff",
        "reproducibility_notes",
        "production_plan",
        "policy_report",
        "execution_trace",
        "task_summary",
        "provenance",
        "reproduction_status",
        "run_ledger_entry",
        "run_ledger",
        "production_evaluation_metrics",
        "production_ablation_matrix",
        "production_error_taxonomy",
        "production_statistical_tests",
        "production_report_summary",
        "production_memory_index",
        "production_memory_runs",
        "production_memory_failures",
        "production_memory_fingerprints",
        "production_memory_outcomes",
        "production_memory_reflections",
    }
    missing_artifacts = sorted(required_artifact_ids - artifact_ids)
    if missing_artifacts:
        errors.append(f"ara-production artifact_manifest.json is missing artifacts: {missing_artifacts}.")

    handoff_detail = artifact_details.get("ara_handoff")
    handoff_payload = None
    if isinstance(handoff_detail, dict) and handoff_detail.get("path"):
        handoff_payload = _read_json(bundle_path / handoff_detail["path"], errors)
    if not isinstance(handoff_payload, dict):
        errors.append("ara-production profile requires a readable ara_handoff artifact.")
    else:
        dispatch = handoff_payload.get("dispatch")
        if not isinstance(dispatch, dict):
            errors.append("ara_handoff.dispatch must be an object for ara-production.")
        else:
            expected_dispatch = {
                "lab_id": "aira",
                "manifest_path": "research_lab.yaml",
                "bundle_type": "aira_result_bundle",
                "validation_profile": ARA_PRODUCTION_VALIDATION_PROFILE,
                "profile": "production-local",
            }
            for key, expected in expected_dispatch.items():
                if dispatch.get(key) != expected:
                    errors.append(f"ara_handoff.dispatch.{key} must be `{expected}` for ara-production.")
            validation_command = dispatch.get("validation_command")
            if validation_command != "python3 -m aira bundles validate <bundle> --profile ara-production --json":
                errors.append("ara_handoff.dispatch.validation_command must use the ara-production profile.")
            allowed_interfaces = dispatch.get("allowed_interfaces")
            if allowed_interfaces != ["research_lab.yaml", "aira_result_bundle"]:
                errors.append(
                    "ara_handoff.dispatch.allowed_interfaces must be "
                    "['research_lab.yaml', 'aira_result_bundle']."
                )

    _check(
        checks,
        "ara_production_profile",
        "pass" if len(errors) == before else "fail",
        "Production-local ARA handoff profile exposes dispatch, evaluation, memory, and bundle gate artifacts.",
    )


def _validate_ara_production_open_profile(
    *,
    metadata: dict[str, Any],
    artifact_ids: set[str],
    errors: list[str],
    checks: list[dict[str, str]],
) -> None:
    before = len(errors)
    manifest = metadata.get("bundle_manifest")
    if not isinstance(manifest, dict):
        errors.append("ara-production-open profile requires a valid bundle_manifest.json object.")
        manifest = {}

    production_runner = manifest.get("production_runner")
    if not isinstance(production_runner, dict):
        errors.append("ara-production-open profile requires bundle_manifest.json.production_runner.")
    elif production_runner.get("profile") != "production-open":
        errors.append("ara-production-open profile requires production_runner.profile to be `production-open`.")

    if manifest.get("deterministic") is not False:
        errors.append("ara-production-open profile requires bundle_manifest.json.deterministic to be false.")
    for key in ("network_required", "external_datasets_required", "gpu_required", "live_model_calls"):
        if manifest.get(key) is not True:
            errors.append(f"ara-production-open profile requires bundle_manifest.json.{key} to be true.")

    required_artifact_ids = {
        "production_plan",
        "policy_report",
        "execution_trace",
        "task_summary",
        "provenance",
        "reproduction_status",
        "run_ledger_entry",
        "run_ledger",
    }
    missing_artifacts = sorted(required_artifact_ids - artifact_ids)
    if missing_artifacts:
        errors.append(f"ara-production-open artifact_manifest.json is missing artifacts: {missing_artifacts}.")

    _check(
        checks,
        "ara_production_open_profile",
        "pass" if len(errors) == before else "fail",
        "Production-open ARA validation profile exposes open experiment policy, execution, provenance, and ledger artifacts.",
    )


def validate_bundle(
    bundle_path: str | Path,
    *,
    profile: str = DEFAULT_VALIDATION_PROFILE,
) -> BundleValidationResult:
    path = Path(bundle_path).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    checks: list[dict[str, str]] = []
    metadata: dict[str, Any] = {}
    files: dict[str, dict[str, Any]] = {}
    bundle_type: str | None = None
    artifact_ids: set[str] = set()
    artifact_details: dict[str, dict[str, Any]] = {}
    manifest_declares_ara_handoff = False
    if profile not in VALIDATION_PROFILES:
        errors.append(f"Unsupported bundle validation profile: {profile}.")

    if not path.exists():
        _check(checks, "bundle_path", "fail", "Bundle path does not exist.")
        return BundleValidationResult(path, False, [f"Bundle path does not exist: {path}"], checks=checks)
    if not path.is_dir():
        _check(checks, "bundle_path", "fail", "Bundle path is not a directory.")
        return BundleValidationResult(path, False, [f"Bundle path is not a directory: {path}"], checks=checks)
    _check(checks, "bundle_path", "pass", "Bundle path exists and is a directory.")

    files = {relative: _file_report(path, relative) for relative in REQUIRED_FILES}
    for relative in REQUIRED_FILES:
        if not files[relative]["present"]:
            errors.append(f"Missing required bundle file: {relative}")
        elif not files[relative]["is_file"]:
            errors.append(f"Required bundle path is not a file: {relative}")
    _check(
        checks,
        "required_files",
        "pass" if all(item["present"] and item["is_file"] for item in files.values()) else "fail",
        "Required bundle files are present." if not errors else "One or more required bundle files are missing.",
    )

    manifest_path = path / "bundle_manifest.json"
    if manifest_path.exists():
        before = len(errors)
        manifest = _read_json(manifest_path, errors)
        if isinstance(manifest, dict):
            bundle_type = _required_string(manifest, "bundle_type", "bundle_manifest.json", errors)
            domain = _required_string(manifest, "domain", "bundle_manifest.json", errors)
            created_at = _required_string(manifest, "created_at", "bundle_manifest.json", errors)
            if bundle_type != "aira_result_bundle":
                errors.append("bundle_manifest.json field `bundle_type` must be `aira_result_bundle`.")
            if domain != "ai_ml":
                errors.append("bundle_manifest.json field `domain` must be `ai_ml`.")
            ara_handoff = manifest.get("ara_handoff")
            if ara_handoff is not None:
                manifest_declares_ara_handoff = True
                if not isinstance(ara_handoff, dict):
                    errors.append("bundle_manifest.json field `ara_handoff` must be an object when present.")
                else:
                    if ara_handoff.get("schema_version") != ARA_HANDOFF_SCHEMA_VERSION:
                        errors.append(
                            f"bundle_manifest.json ara_handoff.schema_version must be `{ARA_HANDOFF_SCHEMA_VERSION}`."
                        )
                    if ara_handoff.get("gate_profile") != ARA_GATE_PROFILE:
                        errors.append(f"bundle_manifest.json ara_handoff.gate_profile must be `{ARA_GATE_PROFILE}`.")
                    artifact = ara_handoff.get("artifact")
                    if not isinstance(artifact, str) or not _safe_relative_path(artifact):
                        errors.append("bundle_manifest.json ara_handoff.artifact must be a safe relative path.")
            metadata["bundle_manifest"] = manifest
            metadata["domain"] = domain or None
            metadata["created_at"] = created_at or None
        elif manifest is not None:
            errors.append("bundle_manifest.json must contain a JSON object.")
        _check(
            checks,
            "bundle_manifest",
            "pass" if len(errors) == before else "fail",
            "bundle_manifest.json declares an AIRA AI/ML result bundle.",
        )

    artifact_manifest_path = path / "artifact_manifest.json"
    if artifact_manifest_path.exists():
        before = len(errors)
        artifact_payload = _read_json(artifact_manifest_path, errors)
        artifact_ids, artifact_details = _validate_artifact_manifest(
            artifact_payload,
            bundle_path=path,
            errors=errors,
        )
        metadata["artifact_ids"] = sorted(artifact_ids)
        metadata["artifact_count"] = len(artifact_ids)
        _check(
            checks,
            "artifact_manifest",
            "pass" if len(errors) == before else "fail",
            "artifact_manifest.json declares resolvable bundle artifacts.",
        )
        provenance_paths = _artifact_paths_for(
            artifact_details,
            artifact_ids={"provenance"},
            kinds={"provenance"},
        )
        if provenance_paths:
            before = len(errors)
            for relative in provenance_paths:
                _validate_provenance_artifact(
                    _read_json(path / relative, errors),
                    relative,
                    errors,
                )
            metadata["provenance_artifacts"] = provenance_paths
            _check(
                checks,
                "provenance_artifacts",
                "pass" if len(errors) == before else "fail",
                "Provenance artifacts declare deterministic local execution inputs.",
            )
        run_ledger_paths = _artifact_paths_for(
            artifact_details,
            artifact_ids={"run_ledger", "run_ledger_entry"},
            kinds={"run_ledger", "run_ledger_entry"},
        )
        if run_ledger_paths:
            before = len(errors)
            run_ids: set[str] = set()
            for relative in run_ledger_paths:
                run_ids.update(_validate_run_ledger_artifact(path / relative, relative, errors))
            metadata["run_ledger_artifacts"] = run_ledger_paths
            metadata["run_ledger_entry_count"] = len(run_ids)
            metadata["run_ledger_run_ids"] = sorted(run_ids)
            _check(
                checks,
                "run_ledger_artifacts",
                "pass" if len(errors) == before else "fail",
                "Run ledger artifacts contain machine-readable experiment memory entries.",
            )
        ara_handoff_paths = _artifact_paths_for(
            artifact_details,
            artifact_ids={"ara_handoff"},
            kinds={"ara_handoff"},
        )
        reproducibility_note_paths = _artifact_paths_for(
            artifact_details,
            artifact_ids={"reproducibility_notes"},
            kinds={"reproducibility_notes"},
        )
        if manifest_declares_ara_handoff or ara_handoff_paths:
            before = len(errors)
            if not ara_handoff_paths:
                errors.append("ARA handoff bundles must declare an ara_handoff artifact.")
            if not reproducibility_note_paths:
                errors.append("ARA handoff bundles must declare reproducibility_notes artifacts.")
            ara_gate_metadata: dict[str, Any] = {
                "profile": None,
                "handoff_artifacts": ara_handoff_paths,
                "reproducibility_note_artifacts": reproducibility_note_paths,
            }
            for relative in ara_handoff_paths:
                ara_gate_metadata.update(
                    _validate_ara_handoff_artifact(
                        _read_json(path / relative, errors),
                        bundle_path=path,
                        artifact_details=artifact_details,
                        errors=errors,
                    )
                )
            for relative in reproducibility_note_paths:
                if not (path / relative).read_text(encoding="utf-8").strip():
                    errors.append(f"{relative} must contain non-empty ARA reproducibility notes.")
            metadata["ara_gate"] = ara_gate_metadata
            _check(
                checks,
                "ara_handoff",
                "pass" if len(errors) == before else "fail",
                "ARA handoff metadata declares bundle, reproduction, claim, and run ledger gate inputs.",
            )

    claims_path = path / "claims.json"
    if claims_path.exists():
        before = len(errors)
        claims_payload = _read_json(claims_path, errors)
        metadata["claim_count"] = _validate_claims(
            claims_payload,
            artifact_ids=artifact_ids,
            artifact_details=artifact_details,
            errors=errors,
            warnings=warnings,
        )
        _check(
            checks,
            "claims",
            "pass" if len(errors) == before else "fail",
            "claims.json satisfies the AIRA reproduction-backed claim contract.",
        )

    for relative in ("writing_brief.md", "limitations.md"):
        file_path = path / relative
        if file_path.exists() and file_path.is_file():
            if file_path.read_text(encoding="utf-8").strip():
                _check(checks, relative, "pass", f"{relative} is non-empty.")
            else:
                warnings.append(f"{relative} is empty.")
                _check(checks, relative, "warn", f"{relative} is empty.")

    if profile == ARA_PRODUCTION_VALIDATION_PROFILE:
        _validate_ara_production_profile(
            bundle_path=path,
            metadata=metadata,
            artifact_ids=artifact_ids,
            artifact_details=artifact_details,
            errors=errors,
            checks=checks,
        )
    elif profile == ARA_PRODUCTION_OPEN_VALIDATION_PROFILE:
        _validate_ara_production_open_profile(
            metadata=metadata,
            artifact_ids=artifact_ids,
            errors=errors,
            checks=checks,
        )

    metadata["required_files"] = list(REQUIRED_FILES)
    metadata["validation_profile"] = profile
    return BundleValidationResult(
        path=path,
        valid=not errors,
        errors=errors,
        warnings=warnings,
        checks=checks,
        files=files,
        metadata=metadata,
        bundle_type=bundle_type,
    )
