"""Inventory legacy ARA AI experiment responsibilities for AIRA migration."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


INVENTORY_SCHEMA_VERSION = "aira.ai_migration_inventory.v1"
CREATED_AT = "2026-05-18T00:00:00Z"

RESPONSIBILITY_CANDIDATES = [
    {
        "id": "experiment_agent",
        "legacy_path": "ara/agents/experiment.py",
        "legacy_responsibility": "Generate, execute, and collect AI/ML experiment artifacts.",
        "aira_target": "aira.production_runner",
        "migration_action": "Split domain benchmark execution from ARA platform orchestration into a profile-gated AIRA runner.",
        "mvp_status": "ported_production_local",
    },
    {
        "id": "code_executor",
        "legacy_path": "ara/tools/code_executor.py",
        "legacy_responsibility": "Run generated experiment code under a local sandbox contract.",
        "aira_target": "aira.production_runner",
        "migration_action": "Keep execution deterministic, bounded, package-controlled, and exposed through result bundles.",
        "mvp_status": "ported_production_local",
    },
    {
        "id": "statistical_tester",
        "legacy_path": "ara/tools/statistical_tester.py",
        "legacy_responsibility": "Evaluate experiment metrics and statistical comparisons.",
        "aira_target": "AIRA evaluation helpers",
        "migration_action": "Port reusable metric checks behind benchmark registry entries.",
        "mvp_status": "inventoried",
    },
    {
        "id": "openml_template",
        "legacy_path": "ara/templates_openml_curated_task01.py",
        "legacy_responsibility": "Curated OpenML experiment template for AI/ML projects.",
        "aira_target": "AIRA dataset and benchmark registries",
        "migration_action": "Represent external datasets as registry entries before enabling downloads.",
        "mvp_status": "placeholder_registry_created",
    },
    {
        "id": "ai_runtime_config",
        "legacy_path": "config.publishable.yaml",
        "legacy_responsibility": "AI/ML runtime, resources, scientific repair, and model configuration.",
        "aira_target": "research_lab.yaml and future AIRA run profiles",
        "migration_action": "Move domain-specific resource and benchmark defaults to AIRA.",
        "mvp_status": "manifest_created",
    },
    {
        "id": "aira_bundle_contract",
        "legacy_path": "ara/labs/bundle_ingest.py",
        "legacy_responsibility": "Shared ARA ecosystem result bundle validation, including AIRA claims.",
        "aira_target": "aira.bundles",
        "migration_action": "Mirror AIRA bundle validation locally for producer-side checks.",
        "mvp_status": "ported_mvp",
    },
    {
        "id": "revision_scientific_repair",
        "legacy_path": "scripts/revise_to_target_score.py",
        "legacy_responsibility": "Plan and run bounded scientific repair experiments for AI papers.",
        "aira_target": "future AIRA experiment planning mode",
        "migration_action": "Inventory only; no live model planning in the bootstrap MVP.",
        "mvp_status": "inventory_only",
    },
]


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_count(path: Path) -> int | None:
    if not path.exists() or not path.is_file():
        return None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


def _read_text(path: Path, limit: int = 500_000) -> str:
    if not path.exists() or not path.is_file():
        return ""
    data = path.read_bytes()[:limit]
    return data.decode("utf-8", errors="replace")


def _config_inventory(source: Path) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for path in sorted(source.glob("config*.yaml")):
        text = _read_text(path)
        research_domain = ""
        for line in text.splitlines():
            if line.strip().startswith("research_domain:"):
                research_domain = line.split(":", 1)[1].strip()
                break
        configs.append(
            {
                "path": path.relative_to(source).as_posix(),
                "research_domain": research_domain,
                "mentions_ai_ml": "ai_ml" in text,
                "mentions_datasets": "datasets" in text,
                "mentions_training": "training" in text.lower(),
                "mentions_scientific_repair": "scientific_repair" in text,
            }
        )
    return configs


def _legacy_project_inventory(source: Path) -> dict[str, Any]:
    projects_dir = source / "projects"
    if not projects_dir.exists() or not projects_dir.is_dir():
        return {"path": "projects", "exists": False, "total_count": 0, "sampled": []}
    projects = sorted(path for path in projects_dir.iterdir() if path.is_dir())
    sampled = []
    for path in projects[:25]:
        sampled.append(
            {
                "name": path.name,
                "has_exp_dir": (path / "exp").exists(),
                "has_writing_dir": (path / "writing").exists(),
                "has_pipeline_results": (path / "pipeline_results.json").exists(),
            }
        )
    return {
        "path": "projects",
        "exists": True,
        "total_count": len(projects),
        "sampled_count": len(sampled),
        "sampled": sampled,
    }


def _responsibility_inventory(source: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for candidate in RESPONSIBILITY_CANDIDATES:
        path = source / candidate["legacy_path"]
        entry = dict(candidate)
        entry.update(
            {
                "exists": path.exists(),
                "line_count": _line_count(path),
                "sha256": _sha256(path),
            }
        )
        entries.append(entry)
    return entries


def build_inventory(source: str | Path) -> dict[str, Any]:
    source_path = Path(source).expanduser().resolve()
    source_exists = source_path.exists() and source_path.is_dir()
    responsibilities = _responsibility_inventory(source_path) if source_exists else []
    configs = _config_inventory(source_path) if source_exists else []
    legacy_projects = _legacy_project_inventory(source_path) if source_exists else {
        "path": "projects",
        "exists": False,
        "total_count": 0,
        "sampled": [],
    }
    present = [item for item in responsibilities if item["exists"]]
    missing = [item for item in responsibilities if not item["exists"]]
    return {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "created_at": CREATED_AT,
        "source": str(source_path),
        "source_exists": source_exists,
        "summary": {
            "responsibility_count": len(responsibilities),
            "present_responsibility_count": len(present),
            "missing_responsibility_count": len(missing),
            "ai_config_count": sum(1 for item in configs if item["research_domain"] == "ai_ml"),
            "legacy_project_count": legacy_projects["total_count"],
            "mvp_targets": [
                "research_lab.yaml",
                "aira.registries",
                "aira.bundles",
                "aira.run-fixture-benchmark",
            ],
            "production_targets": [
                "aira.production_runner",
                "aira.experiments run --profile production-local",
                "tests/fixtures/production_plan.json",
            ],
            "production_runner_status": "production_local_ported",
        },
        "responsibilities": responsibilities,
        "configs": configs,
        "legacy_projects": legacy_projects,
        "notes": [
            "Inventory is read-only and bounded to source, config, and top-level legacy project metadata.",
            "No live model calls, training, downloads, or legacy project writes are performed.",
        ],
    }
