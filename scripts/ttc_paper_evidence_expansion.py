#!/usr/bin/env python3
"""Expanded paper-evidence experiments for the confirmed TTC policy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import ttc_fresh_locked_holdout as fresh_eval
import ttc_pareto_long_loop as long_loop
import ttc_policy_search_loop as search_loop


CONFIRMATION_BUNDLE = Path(
    "/home/biostar/work/projects/aira/runs/ttc_pareto_long_loop_20260522_121029"
)
BASE_SEED = 2026052300
DEFAULT_MODELS = "gpt-4.1-mini,gpt-4.1-nano,gpt-4o-mini"
DEFAULT_VERIFIER_MODEL = os.environ.get("ARA_TTC_VERIFIER_MODEL", "gpt-4.1-mini")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["status"]
        rows = [{"status": "empty"}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def available_models(requested: list[str]) -> tuple[list[str], dict[str, Any]]:
    report = {
        "schema_version": "aira.ttc_model_availability.v1",
        "requested_models": requested,
        "available_models": [],
        "unavailable_models": [],
        "provider_probe": "openai.models.list",
    }
    try:
        from openai import OpenAI

        names = {model.id for model in OpenAI().models.list().data}
    except Exception as exc:  # noqa: BLE001 - recorded in experiment evidence.
        report["probe_error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        report["available_models"] = requested
        return requested, report
    selected = [name for name in requested if name in names]
    missing = [name for name in requested if name not in names]
    report["available_models"] = selected
    report["unavailable_models"] = missing
    if not selected:
        raise SystemExit(f"No requested subject models are available: {requested}")
    return selected, report


def confirmed_policy() -> dict[str, Any]:
    state = read_json(CONFIRMATION_BUNDLE / "artifacts/tasks/pareto_long_loop/final_loop_state.json")
    policy = state["confirmed_policy"]
    return {
        "policy": policy["policy"],
        "family": "gpu_torch_matched_coverage",
        "cost_penalty": float(policy["cost_penalty"]),
        "cheap_delta": float(policy["cheap_delta"]),
        "old_metrics": policy["old_metrics"],
        "old_gate": policy["old_gate"],
    }


def evidence_candidates() -> list[dict[str, Any]]:
    base = confirmed_policy()
    variants = [
        (base["cost_penalty"], base["cheap_delta"], "confirmed"),
        (0.0, 0.0, "no_cost_penalty"),
        (0.015, 0.0, "lower_cost_penalty"),
        (0.02, 0.0025, "nearby_low_penalty_delta"),
        (0.03, 0.0, "nearby_high_penalty"),
        (0.035, 0.0, "higher_cost_penalty"),
        (0.025, 0.005, "cheap_delta_005"),
        (0.025, 0.01, "cheap_delta_010"),
    ]
    rows = []
    seen = set()
    for penalty, cheap_delta, label in variants:
        name = f"{label}_p{penalty:.4f}_d{cheap_delta:.4f}".replace(".", "p")
        if (penalty, cheap_delta) in seen:
            continue
        seen.add((penalty, cheap_delta))
        rows.append(
            {
                "policy": name,
                "label": label,
                "family": "gpu_torch_matched_coverage",
                "cost_penalty": penalty,
                "cheap_delta": cheap_delta,
                "is_confirmed_policy": label == "confirmed",
                "old_metrics": base["old_metrics"] if label == "confirmed" else {},
                "old_gate": base["old_gate"] if label == "confirmed" else {},
            }
        )
    return rows


def materialize_with_items(
    out: Path,
    *,
    items: list[dict[str, Any]],
    model: str,
    seed: int,
    workers: int,
) -> dict[str, Any]:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    stage2 = long_loop.load_module(
        long_loop.EXPERIMENTS_DIR / "run_real_benchmark_stage2.py",
        f"ara_ttc_stage2_evidence_{model.replace('-', '_').replace('.', '_')}_{seed}_{int(time.time())}",
    )
    stage2.SEED = seed
    stage2.MATH_ITEMS = sum(1 for item in items if item["domain"] == "math")
    stage2.FACTUAL_ITEMS = sum(1 for item in items if item["domain"] == "factual")
    stage2.CODE_ITEMS = sum(1 for item in items if item["domain"] == "code")
    stage2.MODEL_ID = model
    stage2.MAX_WORKERS = workers
    stage2.CACHE_PATH = out / "stage2_openai_cache.json"
    stage2.load_items = lambda: items
    cwd = Path.cwd()
    os.chdir(out)
    try:
        stage2.materialize(argparse.Namespace())
    finally:
        os.chdir(cwd)
    return read_json(out / "materialization_summary.json")


def evaluate_topk_baseline(
    rows: list[dict[str, Any]],
    *,
    score_name: str,
    accepted_count: int,
    policy_name: str,
    cost_penalty: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped = fresh_eval.group_items(rows, "holdout")
    selected = []
    for item_id, item_rows in grouped.items():
        scored = []
        for row in item_rows:
            score = fresh_eval.as_float(row.get(score_name)) - cost_penalty * fresh_eval.as_float(row.get("cost"))
            scored.append((score, fresh_eval.as_float(row.get("cost")), row))
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected.append(scored[0])
    selected.sort(key=lambda item: (-item[0], item[1]))
    accepted_ids = {row["item_id"] for _, _, row in selected[:accepted_count]}
    decisions = []
    for score, _, row in selected:
        accepted = row["item_id"] in accepted_ids
        decisions.append(
            {
                "policy": policy_name,
                "split": "holdout",
                "item_id": row["item_id"],
                "domain": row["domain"],
                "action": row["action"] if accepted else "abstain",
                "accepted": accepted,
                "correct": bool(row["correct"]) if accepted else "",
                "cost": fresh_eval.as_float(row.get("cost")) if accepted else 0.0,
                "score": score,
            }
        )
    return decisions, long_loop.metric(decisions, len(grouped))


def domain_metrics(decisions: list[dict[str, Any]], *, prefix: str) -> list[dict[str, Any]]:
    rows = []
    domains = sorted({row["domain"] for row in decisions})
    for domain in ["overall", *domains]:
        subset = decisions if domain == "overall" else [row for row in decisions if row["domain"] == domain]
        rows.append({"policy": prefix, "domain": domain, **long_loop.metric(subset, len(subset))})
    return rows


def evaluate_model_panel(
    out: Path,
    *,
    panel_index: int,
    model: str,
    items: list[dict[str, Any]],
    model_spec: dict[str, Any],
    old_baseline: dict[str, float],
    candidates: list[dict[str, Any]],
    verifier_model: str,
    workers: int,
    verifier_workers: int,
) -> dict[str, Any]:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    seed = BASE_SEED + panel_index
    materialization = materialize_with_items(out, items=items, model=model, seed=seed, workers=workers)
    verifier = long_loop.verify_fresh_round(out, verifier_model=verifier_model, workers=verifier_workers)
    verified_rows = read_jsonl(out / "fresh_verified_response_matrix.jsonl")
    target_coverage = float(old_baseline["coverage"])

    verifier_threshold = fresh_eval.tune_threshold(verified_rows, "tune", "verifier_score", target_coverage)
    confidence_threshold = fresh_eval.tune_threshold(verified_rows, "tune", "confidence", target_coverage)
    verifier_decisions, verifier_metrics = fresh_eval.evaluate_threshold_baseline(
        verified_rows,
        "holdout",
        "verifier_score",
        verifier_threshold,
        "fresh_matched_verifier_threshold",
    )
    confidence_decisions, confidence_metrics = fresh_eval.evaluate_threshold_baseline(
        verified_rows,
        "holdout",
        "confidence",
        confidence_threshold,
        "fresh_matched_confidence_threshold",
    )
    baseline_k = int(verifier_metrics["accepted_count"])
    verifier_topk_decisions, verifier_topk_metrics = evaluate_topk_baseline(
        verified_rows,
        score_name="verifier_score",
        accepted_count=baseline_k,
        policy_name="fresh_verifier_topk_matched_count",
    )
    confidence_topk_decisions, confidence_topk_metrics = evaluate_topk_baseline(
        verified_rows,
        score_name="confidence",
        accepted_count=baseline_k,
        policy_name="fresh_confidence_topk_matched_count",
    )
    cost_aware_topk_decisions, cost_aware_topk_metrics = evaluate_topk_baseline(
        verified_rows,
        score_name="verifier_score",
        accepted_count=baseline_k,
        policy_name="fresh_verifier_cost_aware_topk_p0p025",
        cost_penalty=0.025,
    )

    all_decisions = [
        *verifier_decisions,
        *confidence_decisions,
        *verifier_topk_decisions,
        *confidence_topk_decisions,
        *cost_aware_topk_decisions,
    ]
    metric_rows = [
        {"policy": "fresh_matched_verifier_threshold", **verifier_metrics, "threshold": verifier_threshold},
        {"policy": "fresh_matched_confidence_threshold", **confidence_metrics, "threshold": confidence_threshold},
        {"policy": "fresh_verifier_topk_matched_count", **verifier_topk_metrics},
        {"policy": "fresh_confidence_topk_matched_count", **confidence_topk_metrics},
        {"policy": "fresh_verifier_cost_aware_topk_p0p025", **cost_aware_topk_metrics},
    ]
    candidate_summaries = []
    for candidate in candidates:
        decisions, metrics = long_loop.evaluate_matched_candidate(verified_rows, model_spec, candidate, baseline_k)
        all_decisions.extend(decisions)
        candidate_risk = long_loop.normalize_risk(metrics["risk"])
        verifier_risk = long_loop.normalize_risk(verifier_metrics["risk"])
        verifier_topk_risk = long_loop.normalize_risk(verifier_topk_metrics["risk"])
        gate = {
            "pareto_vs_threshold": (
                metrics["coverage"] >= verifier_metrics["coverage"] - 1e-9
                and candidate_risk <= verifier_risk + 1e-9
                and metrics["cost"] <= verifier_metrics["cost"] + 1e-9
            ),
            "pareto_vs_verifier_topk": (
                metrics["coverage"] >= verifier_topk_metrics["coverage"] - 1e-9
                and candidate_risk <= verifier_topk_risk + 1e-9
                and metrics["cost"] <= verifier_topk_metrics["cost"] + 1e-9
            ),
            "risk_delta_vs_threshold": candidate_risk - verifier_risk,
            "cost_delta_vs_threshold": metrics["cost"] - verifier_metrics["cost"],
            "risk_delta_vs_verifier_topk": candidate_risk - verifier_topk_risk,
            "cost_delta_vs_verifier_topk": metrics["cost"] - verifier_topk_metrics["cost"],
        }
        row = {
            "policy": candidate["policy"],
            "label": candidate["label"],
            "cost_penalty": candidate["cost_penalty"],
            "cheap_delta": candidate["cheap_delta"],
            **metrics,
            **gate,
        }
        candidate_summaries.append(row)
        metric_rows.append(row)

    write_csv(out / "model_panel_metrics.csv", metric_rows)
    write_csv(out / "model_panel_decisions.csv", all_decisions)
    write_csv(
        out / "domain_metrics.csv",
        [
            *domain_metrics(verifier_decisions, prefix="fresh_matched_verifier_threshold"),
            *[
                item
                for candidate in candidates
                if candidate["is_confirmed_policy"]
                for decisions, _ in [long_loop.evaluate_matched_candidate(verified_rows, model_spec, candidate, baseline_k)]
                for item in domain_metrics(decisions, prefix="confirmed_policy")
            ],
        ],
    )
    summary = {
        "schema_version": "aira.ttc_paper_evidence_model_panel.v1",
        "panel_index": panel_index,
        "subject_model": model,
        "verifier_model": verifier_model,
        "item_count": len(items),
        "holdout_count": sum(1 for item in items if item["split"] == "holdout"),
        "materialization": materialization,
        "verifier": verifier,
        "baseline_metrics": {
            "fresh_matched_verifier_threshold": verifier_metrics,
            "fresh_matched_confidence_threshold": confidence_metrics,
            "fresh_verifier_topk_matched_count": verifier_topk_metrics,
            "fresh_confidence_topk_matched_count": confidence_topk_metrics,
            "fresh_verifier_cost_aware_topk_p0p025": cost_aware_topk_metrics,
        },
        "candidate_summaries": candidate_summaries,
    }
    write_json(out / "model_panel_summary.json", summary)
    return summary


def collect_decisions(panel_dirs: list[Path], policy: str, baseline: str) -> list[dict[str, Any]]:
    units = []
    for panel_dir in panel_dirs:
        rows = list(csv.DictReader((panel_dir / "model_panel_decisions.csv").open(encoding="utf-8")))
        by_policy: dict[str, dict[str, dict[str, Any]]] = {}
        for row in rows:
            by_policy.setdefault(row["policy"], {})[row["item_id"]] = row
        ids = sorted(set(by_policy.get(policy, {})) | set(by_policy.get(baseline, {})))
        for item_id in ids:
            p = by_policy.get(policy, {}).get(item_id, {})
            b = by_policy.get(baseline, {}).get(item_id, {})
            units.append(
                {
                    "panel": panel_dir.name,
                    "item_id": item_id,
                    "domain": p.get("domain") or b.get("domain"),
                    "policy_accepted": str(p.get("accepted")).lower() == "true",
                    "policy_correct": str(p.get("correct")).lower() == "true",
                    "policy_cost": search_loop.as_float(p.get("cost")),
                    "baseline_accepted": str(b.get("accepted")).lower() == "true",
                    "baseline_correct": str(b.get("correct")).lower() == "true",
                    "baseline_cost": search_loop.as_float(b.get("cost")),
                }
            )
    return units


def summarize_units(units: list[dict[str, Any]]) -> dict[str, Any]:
    p_acc = [row for row in units if row["policy_accepted"]]
    b_acc = [row for row in units if row["baseline_accepted"]]
    n = len(units)
    p_errors = sum(1 for row in p_acc if not row["policy_correct"])
    b_errors = sum(1 for row in b_acc if not row["baseline_correct"])
    p_risk = p_errors / len(p_acc) if p_acc else math.nan
    b_risk = b_errors / len(b_acc) if b_acc else math.nan
    p_cost = sum(row["policy_cost"] for row in p_acc) / n if n else 0.0
    b_cost = sum(row["baseline_cost"] for row in b_acc) / n if n else 0.0
    return {
        "n_items": n,
        "policy_accepted": len(p_acc),
        "baseline_accepted": len(b_acc),
        "policy_errors": p_errors,
        "baseline_errors": b_errors,
        "policy_coverage": len(p_acc) / n if n else 0.0,
        "baseline_coverage": len(b_acc) / n if n else 0.0,
        "policy_risk": p_risk,
        "baseline_risk": b_risk,
        "policy_cost": p_cost,
        "baseline_cost": b_cost,
        "risk_delta": long_loop.normalize_risk(p_risk) - long_loop.normalize_risk(b_risk),
        "cost_delta": p_cost - b_cost,
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    frac = position - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def bootstrap_summary(units: list[dict[str, Any]], *, seed: int = 20260523, draws: int = 2000) -> dict[str, Any]:
    rng = random.Random(seed)
    risk_deltas = []
    cost_deltas = []
    coverage_deltas = []
    for _ in range(draws):
        sample = [units[rng.randrange(len(units))] for _ in units]
        s = summarize_units(sample)
        risk_deltas.append(float(s["risk_delta"]))
        cost_deltas.append(float(s["cost_delta"]))
        coverage_deltas.append(float(s["policy_coverage"] - s["baseline_coverage"]))
    return {
        "draws": draws,
        "risk_delta_ci95": [percentile(risk_deltas, 0.025), percentile(risk_deltas, 0.975)],
        "risk_delta_median": percentile(risk_deltas, 0.5),
        "cost_delta_ci95": [percentile(cost_deltas, 0.025), percentile(cost_deltas, 0.975)],
        "cost_delta_median": percentile(cost_deltas, 0.5),
        "coverage_delta_ci95": [percentile(coverage_deltas, 0.025), percentile(coverage_deltas, 0.975)],
        "coverage_delta_median": percentile(coverage_deltas, 0.5),
    }


def prior_round03_failure_analysis() -> dict[str, Any]:
    root = CONFIRMATION_BUNDLE / "work/tasks/pareto_long_loop/round_03"
    metrics = list(csv.DictReader((root / "candidate_fresh_metrics.csv").open(encoding="utf-8")))
    confirmed = next(row for row in metrics if row["policy"] == "gpu_torch_matched_coverage_p0p0250_d0p0000")
    decisions = list(csv.DictReader((root / "fresh_policy_decisions.csv").open(encoding="utf-8")))
    rows = []
    for policy in ["gpu_torch_matched_coverage_p0p0250_d0p0000", "fresh_matched_verifier_threshold"]:
        policy_rows = []
        for row in decisions:
            if row["policy"] != policy:
                continue
            policy_rows.append(
                {
                    **row,
                    "accepted": str(row.get("accepted")).lower() == "true",
                    "correct": str(row.get("correct")).lower() == "true",
                    "cost": search_loop.as_float(row.get("cost")),
                }
            )
        rows.extend(domain_metrics_for_rows(policy_rows, policy))
    return {
        "schema_version": "aira.ttc_prior_round03_failure_analysis.v1",
        "source_round": str(root),
        "confirmed_policy_row": confirmed,
        "primary_failure_reason": (
            "The confirmed policy had lower risk than the threshold baseline in round_03, "
            "but its normalized cost was higher by 0.0333, so strict Pareto dominance failed."
        ),
        "domain_metrics": rows,
    }


def domain_metrics_for_rows(decisions: list[dict[str, Any]], policy: str) -> list[dict[str, Any]]:
    return domain_metrics(decisions, prefix=policy)


def run_evidence(args: argparse.Namespace) -> None:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    requested_models = [model.strip() for model in args.models.split(",") if model.strip()]
    models, availability = available_models(requested_models)
    write_json(out / "model_availability.json", availability)

    candidates = evidence_candidates()
    _, _, old_baseline, model_spec = long_loop.build_candidate_pool()
    excluded = set(long_loop.prior_fingerprints())
    confirmation_state = read_json(CONFIRMATION_BUNDLE / "artifacts/tasks/pareto_long_loop/final_loop_state.json")
    excluded.update(confirmation_state.get("excluded_fingerprints", []))

    manifest = []
    all_summaries = []
    start = time.monotonic()
    for panel_index in range(1, args.panels + 1):
        if (time.monotonic() - start) / 3600.0 >= args.time_budget_hours:
            break
        seed = BASE_SEED + panel_index
        panel_items = long_loop.load_unique_items(
            seed=seed,
            math_items=args.math_items,
            factual_items=args.factual_items,
            code_items=args.code_items,
            excluded=excluded,
        )
        fingerprints = [long_loop.item_fingerprint(item) for item in panel_items]
        overlap = sorted(set(fingerprints) & excluded)
        panel_audit = {
            "schema_version": "aira.ttc_paper_evidence_panel_audit.v1",
            "panel_index": panel_index,
            "seed": seed,
            "item_count": len(panel_items),
            "holdout_count": sum(1 for item in panel_items if item["split"] == "holdout"),
            "overlap_count": len(overlap),
            "zero_overlap": len(overlap) == 0,
            "domain_counts": {
                domain: sum(1 for item in panel_items if item["domain"] == domain)
                for domain in sorted({item["domain"] for item in panel_items})
            },
        }
        panel_dir = out / f"panel_{panel_index:02d}"
        panel_dir.mkdir(parents=True, exist_ok=True)
        write_json(panel_dir / "panel_overlap_audit.json", panel_audit)
        write_json(panel_dir / "dataset_items.json", {"schema_version": "aira.ttc.paper_evidence_panel_items.v1", "items": panel_items})
        if overlap:
            raise RuntimeError(f"Panel {panel_index} was not zero-overlap.")

        for model in models:
            model_dir = panel_dir / model.replace("/", "_")
            summary = evaluate_model_panel(
                model_dir,
                panel_index=panel_index,
                model=model,
                items=panel_items,
                model_spec=model_spec,
                old_baseline=old_baseline,
                candidates=candidates,
                verifier_model=args.verifier_model,
                workers=args.workers,
                verifier_workers=args.verifier_workers,
            )
            all_summaries.append(summary)
            manifest.append(
                {
                    "panel_index": panel_index,
                    "subject_model": model,
                    "path": str(model_dir.relative_to(out)),
                    "status": "passed",
                    "confirmed_policy_threshold_pareto": next(
                        item
                        for item in summary["candidate_summaries"]
                        if item["label"] == "confirmed"
                    )["pareto_vs_threshold"],
                    "confirmed_policy_topk_pareto": next(
                        item
                        for item in summary["candidate_summaries"]
                        if item["label"] == "confirmed"
                    )["pareto_vs_verifier_topk"],
                }
            )
        excluded.update(fingerprints)

    write_json(out / "round_artifact_manifest.json", {"schema_version": "aira.ttc_paper_evidence_manifest.v1", "runs": manifest})

    model_dirs = [out / item["path"] for item in manifest]
    confirmed_policy_name = next(candidate["policy"] for candidate in candidates if candidate["is_confirmed_policy"])
    stat_rows = []
    stat_payload: dict[str, Any] = {"schema_version": "aira.ttc_paper_evidence_statistics.v1", "groups": []}
    for group_name, dirs in [
        ("overall", model_dirs),
        *[(model, [out / item["path"] for item in manifest if item["subject_model"] == model]) for model in models],
    ]:
        if not dirs:
            continue
        for baseline in ["fresh_matched_verifier_threshold", "fresh_verifier_topk_matched_count", "fresh_verifier_cost_aware_topk_p0p025"]:
            units = collect_decisions(dirs, confirmed_policy_name, baseline)
            summary = summarize_units(units)
            boot = bootstrap_summary(units, draws=args.bootstrap_draws)
            row = {"group": group_name, "baseline": baseline, **summary, **boot}
            stat_payload["groups"].append(row)
            stat_rows.append(row)
    write_json(out / "expanded_statistical_summary.json", stat_payload)
    write_csv(out / "expanded_statistical_summary.csv", stat_rows)

    baseline_rows = []
    for item in manifest:
        summary = read_json(out / item["path"] / "model_panel_summary.json")
        for name, metrics in summary["baseline_metrics"].items():
            baseline_rows.append({"panel_index": item["panel_index"], "subject_model": item["subject_model"], "policy": name, **metrics})
        for candidate in summary["candidate_summaries"]:
            baseline_rows.append({"panel_index": item["panel_index"], "subject_model": item["subject_model"], **candidate})
    write_csv(out / "baseline_ablation_summary.csv", baseline_rows)

    failure = prior_round03_failure_analysis()
    new_failures = [
        item
        for item in manifest
        if not item["confirmed_policy_threshold_pareto"] or not item["confirmed_policy_topk_pareto"]
    ]
    failure["expanded_failure_cases"] = new_failures
    write_json(out / "failure_analysis.json", failure)
    (out / "failure_analysis.md").write_text(
        "\n".join(
            [
                "# Failure Analysis",
                "",
                "## Prior Round 03",
                failure["primary_failure_reason"],
                "",
                "## Expanded Evidence Failures",
                json.dumps(new_failures, indent=2, sort_keys=True),
                "",
            ]
        ),
        encoding="utf-8",
    )

    overall_threshold = next(
        row for row in stat_rows if row["group"] == "overall" and row["baseline"] == "fresh_matched_verifier_threshold"
    )
    threshold_passes = sum(1 for item in manifest if item["confirmed_policy_threshold_pareto"])
    topk_passes = sum(1 for item in manifest if item["confirmed_policy_topk_pareto"])
    claim_gate = {
        "schema_version": "aira.ttc_paper_evidence_claim_gate.v1",
        "status": (
            "expanded_evidence_passed"
            if threshold_passes == len(manifest)
            and overall_threshold["risk_delta_ci95"][1] <= 0
            and overall_threshold["cost_delta_ci95"][1] <= 0
            else "expanded_evidence_mixed"
        ),
        "subject_models": models,
        "panels_completed": len({item["panel_index"] for item in manifest}),
        "model_panel_runs": len(manifest),
        "confirmed_policy": confirmed_policy_name,
        "threshold_pareto_passes": threshold_passes,
        "verifier_topk_pareto_passes": topk_passes,
        "overall_vs_threshold": overall_threshold,
        "interpretation": (
            "Expanded evidence supports the confirmed policy against the strong tune-calibrated threshold baseline."
            if threshold_passes == len(manifest)
            else "Expanded evidence is mixed; inspect failure_analysis and stronger-baseline ablations before manuscript claims."
        ),
    }
    write_json(out / "expanded_claim_gate.json", claim_gate)
    (out / "paper_evidence_report.md").write_text(
        "\n".join(
            [
                "# TTC Paper Evidence Expansion",
                "",
                f"Status: {claim_gate['status']}",
                f"Confirmed policy: `{confirmed_policy_name}`",
                f"Subject models: {', '.join(models)}",
                f"Panels completed: {claim_gate['panels_completed']}",
                f"Model-panel runs: {claim_gate['model_panel_runs']}",
                f"Threshold Pareto passes: {threshold_passes}/{len(manifest)}",
                f"Verifier top-k Pareto passes: {topk_passes}/{len(manifest)}",
                "",
                "## Overall Vs Threshold",
                json.dumps(overall_threshold, indent=2, sort_keys=True),
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run-evidence"])
    parser.add_argument("--out", default=".")
    parser.add_argument("--time-budget-hours", type=float, default=24.0)
    parser.add_argument("--panels", type=int, default=3)
    parser.add_argument("--math-items", type=int, default=96)
    parser.add_argument("--factual-items", type=int, default=96)
    parser.add_argument("--code-items", type=int, default=48)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--verifier-model", default=DEFAULT_VERIFIER_MODEL)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--verifier-workers", type=int, default=8)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    if args.command == "run-evidence":
        run_evidence(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
