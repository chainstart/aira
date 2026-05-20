"""Local research_lab.yaml loader and validator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = "aira.research_lab_manifest.v1"
DEFAULT_MANIFEST_PATH = Path("research_lab.yaml")


@dataclass(frozen=True)
class ManifestValidation:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LabManifest:
    path: Path
    raw: dict[str, Any]
    validation: ManifestValidation

    @property
    def lab_id(self) -> str:
        return str(self.raw.get("lab_id") or "")

    @property
    def domain(self) -> str:
        return str(self.raw.get("domain") or "")

    @property
    def bundle_types(self) -> list[str]:
        bundles = _mapping(self.raw.get("bundles"))
        return _string_list(bundles.get("produced"))

    def to_dict(self) -> dict[str, Any]:
        entrypoints = _mapping(self.raw.get("entrypoints"))
        commands = _mapping(self.raw.get("commands"))
        artifacts = _mapping(self.raw.get("artifacts"))
        safety = _mapping(self.raw.get("safety"))
        registries = _mapping(self.raw.get("registries"))
        legacy = _mapping(self.raw.get("legacy"))
        bundles = _mapping(self.raw.get("bundles"))
        profiles = _mapping(self.raw.get("profiles"))
        return {
            "schema_version": SCHEMA_VERSION,
            "path": str(self.path),
            "lab_id": self.lab_id,
            "system_name": str(self.raw.get("system_name") or ""),
            "full_name": str(self.raw.get("full_name") or ""),
            "domain": self.domain,
            "legacy": {
                "repo_names": _string_list(legacy.get("repo_names")),
                "lab_ids": _string_list(legacy.get("lab_ids")),
            },
            "entrypoints": {
                "agent_cli": _string_list(entrypoints.get("agent_cli")),
                "direct_tools": _string_list(entrypoints.get("direct_tools")),
            },
            "commands": {
                "allow_prefixes": _string_list(commands.get("allow_prefixes")),
                "deny_patterns": _string_list(commands.get("deny_patterns")),
                "smoke": commands.get("smoke") if isinstance(commands.get("smoke"), dict) else None,
            },
            "artifact_globs": {
                "include": _string_list(artifacts.get("include")),
                "exclude": _string_list(artifacts.get("exclude")),
            },
            "bundle_types": self.bundle_types,
            "bundle_handoff_profiles": bundles.get("handoff_profiles")
            if isinstance(bundles.get("handoff_profiles"), list)
            else [],
            "registries": dict(registries),
            "profiles": dict(profiles),
            "safety": dict(safety),
            "valid": self.validation.valid,
            "errors": list(self.validation.errors),
            "warnings": list(self.validation.warnings),
        }


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _require_string(raw: dict[str, Any], key: str, errors: list[str]) -> None:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"`{key}` is required and must be a non-empty string.")


def _validate_string_list(raw: dict[str, Any], key: str, errors: list[str]) -> list[str]:
    value = raw.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        errors.append(f"`{key}` must be a list.")
        return []
    values = _string_list(value)
    if len(values) != len(value):
        errors.append(f"`{key}` must contain only non-empty strings.")
    return values


def validate_manifest(raw: dict[str, Any]) -> ManifestValidation:
    errors: list[str] = []
    warnings: list[str] = []
    for key in ("lab_id", "system_name", "full_name", "domain"):
        _require_string(raw, key, errors)

    entrypoints = _mapping(raw.get("entrypoints"))
    if not entrypoints:
        errors.append("`entrypoints` is required and must be a mapping.")
    if not _validate_string_list(entrypoints, "agent_cli", errors):
        warnings.append("`entrypoints.agent_cli` is empty; ARA cannot dispatch this lab.")
    _validate_string_list(entrypoints, "direct_tools", errors)

    commands = _mapping(raw.get("commands"))
    if not commands:
        errors.append("`commands` is required and must be a mapping.")
    if not _validate_string_list(commands, "allow_prefixes", errors):
        errors.append("`commands.allow_prefixes` must include at least one allowed command prefix.")
    _validate_string_list(commands, "deny_patterns", errors)

    artifacts = _mapping(raw.get("artifacts"))
    if not _validate_string_list(artifacts, "include", errors):
        warnings.append("`artifacts.include` is empty; no bundle artifacts will be discoverable.")
    _validate_string_list(artifacts, "exclude", errors)

    bundles = _mapping(raw.get("bundles"))
    produced = _validate_string_list(bundles, "produced", errors)
    if "aira_result_bundle" not in produced:
        errors.append("`bundles.produced` must include `aira_result_bundle`.")

    safety = _mapping(raw.get("safety"))
    if safety.get("destructive_commands") is not False:
        errors.append("`safety.destructive_commands` must be false.")
    if safety.get("network_policy") not in {"none", "restricted", "unrestricted"}:
        errors.append("`safety.network_policy` must be `none`, `restricted`, or `unrestricted`.")
    if not isinstance(safety.get("live_model_calls"), bool):
        errors.append("`safety.live_model_calls` must be a boolean.")

    return ManifestValidation(valid=not errors, errors=errors, warnings=warnings)


def load_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> LabManifest:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.exists():
        validation = ManifestValidation(False, [f"Manifest not found: {manifest_path}"])
        return LabManifest(path=manifest_path, raw={}, validation=validation)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        validation = ManifestValidation(False, ["Manifest root must be a mapping."])
        return LabManifest(path=manifest_path, raw={}, validation=validation)
    return LabManifest(path=manifest_path, raw=payload, validation=validate_manifest(payload))
