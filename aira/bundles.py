"""AIRA result bundle writer and validator."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any


BUNDLE_SCHEMA_VERSION = "ara.result_bundle.v1"
VALIDATION_SCHEMA_VERSION = "aira.bundle_validation.v1"
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


def validate_bundle(bundle_path: str | Path) -> BundleValidationResult:
    path = Path(bundle_path).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    checks: list[dict[str, str]] = []
    metadata: dict[str, Any] = {}
    files: dict[str, dict[str, Any]] = {}
    bundle_type: str | None = None
    artifact_ids: set[str] = set()
    artifact_details: dict[str, dict[str, Any]] = {}

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

    metadata["required_files"] = list(REQUIRED_FILES)
    metadata["validation_profile"] = "aira-mvp"
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
