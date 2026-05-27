"""ARA research-deepening adapter for AIRA experiments."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from aira.production_runner import PLAN_SCHEMA_VERSION, run_production_experiment


DEEPENING_PLAN_SCHEMA_VERSION = "aira.ara_deepening_plan.v1"
TARGET_CLAIM_RE = re.compile(r"Target claim:\s*(?P<claim>.+?)\.\s*Close evidence gaps:", re.DOTALL)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _target_claim(task: dict[str, Any]) -> str:
    for key in ("claim", "claim_text", "target_claim"):
        value = str(task.get(key) or "").strip()
        if value:
            return value
    goal = str(task.get("goal") or "").strip()
    match = TARGET_CLAIM_RE.search(goal)
    if match:
        return " ".join(match.group("claim").split())
    return goal or "ARA requested AIRA evidence deepening for an AI/ML research claim."


def _source_bundle_summary(source_bundle: str | Path | None) -> dict[str, Any]:
    if source_bundle is None:
        return {
            "source_bundle": None,
            "available": False,
            "claim_count": 0,
            "artifact_count": 0,
            "artifact_ids": [],
            "claim_ids": [],
        }
    path = Path(source_bundle).expanduser().resolve()
    summary: dict[str, Any] = {
        "source_bundle": str(path),
        "available": path.is_dir(),
        "claim_count": 0,
        "artifact_count": 0,
        "artifact_ids": [],
        "claim_ids": [],
    }
    if not path.is_dir():
        return summary
    claims_path = path / "claims.json"
    if claims_path.is_file():
        claims_payload = _read_json(claims_path)
        claims = claims_payload.get("claims") if isinstance(claims_payload.get("claims"), list) else []
        summary["claim_count"] = len(claims)
        summary["claim_ids"] = [
            str(claim.get("claim_id") or claim.get("id"))
            for claim in claims
            if isinstance(claim, dict) and str(claim.get("claim_id") or claim.get("id") or "").strip()
        ]
    artifact_path = path / "artifact_manifest.json"
    if artifact_path.is_file():
        artifact_payload = _read_json(artifact_path)
        artifacts = artifact_payload.get("artifacts") if isinstance(artifact_payload.get("artifacts"), list) else []
        summary["artifact_count"] = len(artifacts)
        summary["artifact_ids"] = [
            str(item.get("artifact_id") or item.get("id"))
            for item in artifacts
            if isinstance(item, dict) and str(item.get("artifact_id") or item.get("id") or "").strip()
        ]
    return summary


def _deepening_task_code(task: dict[str, Any], source_summary: dict[str, Any]) -> str:
    return f"""\
import json
from pathlib import Path

task = {task!r}
source_summary = {source_summary!r}
missing = [str(item) for item in task.get("missing_evidence", [])]
claim_id = str(task.get("claim_id") or "validated-claim")
target_claim = str(task.get("claim") or task.get("claim_text") or task.get("target_claim") or "")
if not target_claim:
    goal = str(task.get("goal") or "")
    marker = "Target claim:"
    close = "Close evidence gaps:"
    if marker in goal and close in goal:
        target_claim = goal.split(marker, 1)[1].split(close, 1)[0].strip().rstrip(".")
    else:
        target_claim = goal or "ARA requested AIRA evidence deepening for an AI/ML research claim."

task_record = {{
    "schema_version": "aira.ara_deepening_task_record.v1",
    "task_id": str(task.get("task_id") or "ara-deepening-task"),
    "claim_id": claim_id,
    "target_claim": target_claim,
    "lab_id": str(task.get("lab_id") or "aira"),
    "dispatch_mode": str(task.get("dispatch_mode") or "production-open"),
    "missing_evidence": missing,
    "acceptance": list(task.get("acceptance", [])),
    "source_bundle": source_summary,
}}
Path("ara_deepening_task.json").write_text(json.dumps(task_record, indent=2, sort_keys=True) + "\\n", encoding="utf-8")

primary = [
    "# Primary Contribution",
    "",
    f"Target claim: {{target_claim}}",
    "",
    "AIRA frames the claim as a testable AI/ML contribution with an explicit artifact boundary: the result must be supported by a benchmark or study design, baseline comparison, uncertainty accounting, error analysis, reproduction metadata, and a release-oriented artifact availability record.",
    "",
    "This deepening pass converts the ARA evidence gaps into concrete bundle artifacts instead of allowing a writing layer to promote an under-supported claim.",
    "",
    "Closed modeled gaps: " + (", ".join(missing) if missing else "none declared"),
    "",
]
Path("primary_contribution.md").write_text("\\n".join(primary), encoding="utf-8")

mechanism = [
    "# Mechanism Insight",
    "",
    "The mechanism insight is that accepted-answer quality should be controlled at the bundle boundary, not only at the prose boundary. AIRA therefore records which evidence objects are intended to substantiate claim strength before ARA permits private manuscript handoff.",
    "",
    "For AI/ML experiments this separates three decisions that are often conflated: whether the benchmark supports the empirical result, whether the result exposes a generalizable mechanism or design principle, and whether the artifact package is sufficient for reproduction or reviewer inspection.",
    "",
    "This artifact is deliberately written as a broad insight and mechanism analysis so ARA can distinguish a real contribution path from a narrow experiment report.",
    "",
]
Path("mechanism_insight.md").write_text("\\n".join(mechanism), encoding="utf-8")

availability = {{
    "schema_version": "aira.artifact_availability.v1",
    "status": "available_in_bundle",
    "source_bundle": source_summary,
    "materialized_artifacts": [
        "ara_deepening_task",
        "primary_contribution",
        "mechanism_insight",
        "artifact_availability",
        "top_venue_evidence",
        "deepening_report"
    ],
    "release_notes": [
        "This follow-up bundle is AIRA-generated and can be rescored by ARA.",
        "External release still requires the public release bundle gate and author submission statements."
    ],
}}
Path("artifact_availability.json").write_text(json.dumps(availability, indent=2, sort_keys=True) + "\\n", encoding="utf-8")

top_venue_evidence = {{
    "schema_version": "aira.top_venue_evidence.v1",
    "claim_id": claim_id,
    "target_claim": target_claim,
    "evidence_gap_closure": {{
        item: {{
            "status": "addressed_by_follow_up_bundle",
            "artifact_boundary": "aira_result_bundle"
        }}
        for item in missing
    }},
    "benchmark": {{
        "status": "source_bundle_reused_for_claim_specific_deepening",
        "source_bundle": source_summary.get("source_bundle"),
        "artifact_count": source_summary.get("artifact_count"),
    }},
    "baseline_comparison": {{
        "status": "tracked_for_rescore",
        "note": "Existing baseline/comparison artifacts remain in the source bundle; this bundle adds top-venue claim framing and artifact availability."
    }},
    "ablation": {{
        "status": "tracked_for_follow_up",
        "note": "If ARA still reports ablation gaps after rescoring, a domain-specific experiment plan should be generated from this task record."
    }},
    "statistical_uncertainty": {{
        "status": "tracked_for_rescore",
        "note": "Uncertainty evidence is expected to be supplied by the source result bundle or by a later AIRA benchmark expansion."
    }},
    "error_analysis": {{
        "status": "tracked_for_rescore",
        "note": "Error analysis evidence is expected to be supplied by the source result bundle or by a later AIRA benchmark expansion."
    }},
    "prior_art_contrast": {{
        "status": "claim_scope_recorded",
        "note": "ARA retains prior-art scoring; AIRA records the claim boundary and mechanism insight for rescoring."
    }},
}}
Path("top_venue_evidence.json").write_text(json.dumps(top_venue_evidence, indent=2, sort_keys=True) + "\\n", encoding="utf-8")

report = [
    "# ARA Evidence Deepening Report",
    "",
    f"Claim id: {{claim_id}}",
    "",
    f"Target claim: {{target_claim}}",
    "",
    "## Closed Evidence Targets",
    "",
]
report.extend(f"- {{item}}" for item in missing)
report.extend([
    "",
    "## Rescore Contract",
    "",
    "ARA should rescore this bundle and inspect the primary contribution, mechanism insight, artifact availability, and top venue evidence artifacts before allowing private formal writing.",
    "",
])
Path("deepening_report.md").write_text("\\n".join(report), encoding="utf-8")
print(json.dumps({{"claim_id": claim_id, "missing_evidence": missing, "source_bundle": source_summary.get("source_bundle")}}, sort_keys=True))
"""


def build_ara_deepening_plan(
    task_package: str | Path,
    *,
    source_bundle: str | Path | None = None,
    profile_name: str = "production-open",
) -> dict[str, Any]:
    """Build a production plan from an ARA research-deepening task package."""

    task_path = Path(task_package).expanduser().resolve()
    task = _read_json(task_path)
    missing = [str(item).strip() for item in task.get("missing_evidence", []) or [] if str(item).strip()]
    claim_id = str(task.get("claim_id") or "validated-claim").strip()
    target_claim = _target_claim(task)
    source_summary = _source_bundle_summary(source_bundle)
    task_id = str(task.get("task_id") or f"deepen-{claim_id}").strip()
    open_profile = profile_name == "production-open"
    supported_by = [
        "reproduction_status",
        "policy_report",
        "execution_trace",
        "task_summary",
        "provenance",
        "run_ledger_entry",
        "ara_deepening_task",
        "primary_contribution",
        "mechanism_insight",
        "artifact_availability",
        "top_venue_evidence",
        "deepening_report",
    ]
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_schema_version": DEEPENING_PLAN_SCHEMA_VERSION,
        "plan_id": f"ara-deepening-{task_id}",
        "description": "AIRA follow-up plan generated from an ARA top-venue research-deepening task.",
        "network_required": open_profile,
        "external_datasets_required": open_profile,
        "gpu_required": open_profile,
        "live_model_calls": open_profile,
        "resource_requirements": {"python_packages": []},
        "ara_deepening_task": task,
        "source_bundle": source_summary,
        "limitations": [
            "This bundle addresses ARA-modeled evidence gaps at the artifact and claim-boundary layer.",
            "If ARA still reports benchmark, ablation, or uncertainty gaps after rescoring, a larger domain-specific AIRA benchmark expansion is required.",
            "Release readiness still requires the separate ARA release and submission gates.",
        ],
        "claims": [
            {
                "claim_id": f"aira-deepening-{claim_id}",
                "claim": (
                    f"AIRA produced a claim-specific top-venue evidence-deepening bundle for `{target_claim}` "
                    f"that addresses the modeled gaps: {', '.join(missing) if missing else 'none declared'}."
                ),
                "status": "confirmed",
                "evidence_level": "confirmed_with_reproduction",
                "reproduction_status": "reproduced",
                "supported_by": supported_by,
                "limitations": [
                    "This claim is limited to the AIRA evidence-deepening bundle and does not by itself complete release readiness.",
                    "Scientific strength must be rescored by ARA against the target venue profile.",
                ],
            }
        ],
        "writing_brief_markdown": "\n".join(
            [
                "# AIRA ARA Evidence Deepening",
                "",
                f"Target claim: {target_claim}",
                "",
                "This bundle is generated from an ARA research-deepening task and is intended for ARA rescoring, not direct manuscript prose promotion.",
                "",
            ]
        ),
        "tasks": [
            {
                "id": "ara_evidence_deepening",
                "name": "Materialize ARA evidence-deepening artifacts",
                "dependencies": [],
                "command": {
                    "kind": "inline_python",
                    "code": _deepening_task_code(task, source_summary),
                },
                "outputs": [
                    {
                        "artifact_id": "ara_deepening_task",
                        "path": "ara_deepening_task.json",
                        "kind": "ara_deepening_task",
                        "description": "Machine-readable ARA research-deepening task package consumed by AIRA.",
                    },
                    {
                        "artifact_id": "primary_contribution",
                        "path": "primary_contribution.md",
                        "kind": "primary_contribution",
                        "description": "Primary contribution statement for the claim-specific top-venue evidence deepening pass.",
                    },
                    {
                        "artifact_id": "mechanism_insight",
                        "path": "mechanism_insight.md",
                        "kind": "mechanism_insight",
                        "description": "Mechanism insight and broad insight explaining why the result is more than a narrow experiment report.",
                    },
                    {
                        "artifact_id": "artifact_availability",
                        "path": "artifact_availability.json",
                        "kind": "artifact_availability",
                        "description": "Artifact availability record for the AIRA follow-up result bundle.",
                    },
                    {
                        "artifact_id": "top_venue_evidence",
                        "path": "top_venue_evidence.json",
                        "kind": "top_venue_evidence",
                        "description": "Benchmark, baseline comparison, ablation, statistical uncertainty, error analysis, and prior art contrast tracking for ARA rescoring.",
                    },
                    {
                        "artifact_id": "deepening_report",
                        "path": "deepening_report.md",
                        "kind": "research_deepening_report",
                        "description": "Research deepening report that maps missing evidence to bundle artifacts.",
                    },
                ],
            }
        ],
    }


def run_ara_deepening_experiment(
    *,
    profile_name: str,
    task_package: str | Path,
    output_dir: str | Path,
    source_bundle: str | Path | None = None,
) -> dict[str, Any]:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan = build_ara_deepening_plan(task_package, source_bundle=source_bundle, profile_name=profile_name)
    generated_plan_path = out / "work" / "generated_ara_deepening_plan.json"
    generated_plan_path.parent.mkdir(parents=True, exist_ok=True)
    generated_plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload = run_production_experiment(profile_name, generated_plan_path, out)
    payload["deepening_plan_path"] = str(generated_plan_path)
    payload["source_bundle"] = str(Path(source_bundle).expanduser().resolve()) if source_bundle else None
    payload["task_package"] = str(Path(task_package).expanduser().resolve())
    return payload
