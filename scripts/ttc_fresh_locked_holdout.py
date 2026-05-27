#!/usr/bin/env python3
"""Fresh locked-holdout validation for the TTC cost-compressed policy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ttc_policy_search_loop as search_loop


PROJECT_ROOT = Path(
    "/home/biostar/work/projects/ara-manuscript-private/"
    "ara_manuscript_private/workspaces/conformal-ttc-risk-control-20260521"
)
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
PRIOR_POLICY_BUNDLE = Path(
    "/home/biostar/work/projects/aira/runs/ttc_policy_cost_compression_20260522_090913"
)
PRIOR_STAGE2_BUNDLE = PROJECT_ROOT / "aira_runs/stage2_real_large_bundle"
FRESH_SEED = 20260522
DEFAULT_MODEL = os.environ.get("ARA_TTC_MODEL", "gpt-4.1-mini")
DEFAULT_VERIFIER_MODEL = os.environ.get("ARA_TTC_VERIFIER_MODEL", "gpt-4.1-mini")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def dep_dir(task_id: str) -> Path:
    deps = json.loads(os.environ.get("AIRA_DEP_DIRS", "{}"))
    if task_id not in deps:
        raise SystemExit(f"Missing dependency task dir for {task_id}")
    return Path(deps[task_id])


def item_fingerprint(item: dict[str, Any]) -> str:
    return sha256_text(json.dumps({
        "domain": item.get("domain"),
        "question": item.get("question"),
        "gold": item.get("gold"),
        "choices": item.get("choices"),
        "public_tests": item.get("public_tests"),
    }, sort_keys=True, ensure_ascii=True))


def lock_policy(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = json.loads(
        (PRIOR_POLICY_BUNDLE / "artifacts/tasks/search_policies/best_policy_spec.json").read_text(encoding="utf-8")
    )
    model = json.loads(
        (PRIOR_POLICY_BUNDLE / "artifacts/tasks/search_policies/torch_meta_model.json").read_text(encoding="utf-8")
    )
    policy_spec = best["threshold_specs"][0]
    baseline = best["baseline"]
    frozen = {
        "schema_version": "aira.ttc_frozen_policy.v1",
        "created_at": now_iso(),
        "source_bundle": str(PRIOR_POLICY_BUNDLE),
        "selected_policy": best["selected_policy"],
        "policy_family": "gpu_torch_tune_budget_cost_compressed",
        "cost_penalty": float(policy_spec["cost_penalty"]),
        "cheap_delta": float(policy_spec["cheap_delta"]),
        "target_coverage": float(baseline["coverage"]),
        "target_baseline": baseline,
        "model_spec": model,
        "decision_rule": (
            "For each fresh holdout item, score verified actions with the frozen torch meta-risk scorer "
            "minus cost_penalty times normalized action cost, choose the top action per item, and accept "
            "ceil(target_coverage * holdout_item_count) items by score. No fresh holdout labels are used."
        ),
    }
    write_json(out / "frozen_policy.json", frozen)
    write_json(out / "gpu_resource_report.json", search_loop.gpu_report())
    write_json(
        out / "fresh_lock_manifest.json",
        {
            "schema_version": "aira.ttc_fresh_lock_manifest.v1",
            "fresh_seed": FRESH_SEED,
            "policy_locked_before_fresh_data": True,
            "policy_source_bundle": str(PRIOR_POLICY_BUNDLE),
            "prior_policy_status": "candidate_found_pareto",
            "fresh_evaluation_rule": "fresh tune may calibrate baselines; candidate policy is frozen before fresh holdout scoring.",
        },
    )


def materialize(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stage2 = load_module(EXPERIMENTS_DIR / "run_real_benchmark_stage2.py", "ara_ttc_stage2_fresh")
    stage2.SEED = args.seed
    stage2.MATH_ITEMS = args.math_items
    stage2.FACTUAL_ITEMS = args.factual_items
    stage2.CODE_ITEMS = args.code_items
    stage2.MODEL_ID = args.model
    stage2.MAX_WORKERS = args.workers
    stage2.CACHE_PATH = out / "fresh_stage2_openai_cache.json"

    cwd = Path.cwd()
    os.chdir(out)
    try:
        stage2.materialize(argparse.Namespace())
    finally:
        os.chdir(cwd)

    items_payload = json.loads((out / "dataset_items.json").read_text(encoding="utf-8"))
    fresh_items = items_payload["items"]
    prior_items_path = PRIOR_STAGE2_BUNDLE / "artifacts/tasks/materialize_real_outputs/dataset_items.json"
    prior_fingerprints: set[str] = set()
    if prior_items_path.exists():
        prior_payload = json.loads(prior_items_path.read_text(encoding="utf-8"))
        prior_fingerprints = {item_fingerprint(item) for item in prior_payload.get("items", [])}
    fresh_fingerprints = {item_fingerprint(item) for item in fresh_items}
    overlap = sorted(fresh_fingerprints & prior_fingerprints)
    write_json(
        out / "fresh_overlap_audit.json",
        {
            "schema_version": "aira.ttc_fresh_overlap_audit.v1",
            "fresh_seed": args.seed,
            "fresh_item_count": len(fresh_items),
            "prior_item_count": len(prior_fingerprints),
            "overlap_count": len(overlap),
            "overlap_fingerprints": overlap[:25],
            "locked_holdout_count": sum(1 for item in fresh_items if item.get("split") == "holdout"),
            "domain_counts": {
                domain: sum(1 for item in fresh_items if item.get("domain") == domain)
                for domain in sorted({item.get("domain") for item in fresh_items})
            },
        },
    )


def verify(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = dep_dir("materialize_fresh_holdout")
    stage3 = load_module(EXPERIMENTS_DIR / "run_verifier_stage3.py", "ara_ttc_stage3_fresh")
    stage3.MODEL_ID = args.verifier_model
    stage3.MAX_WORKERS = args.workers
    stage3.CACHE_PATH = out / "fresh_stage3_verifier_cache.json"

    item_payload = json.loads((source / "dataset_items.json").read_text(encoding="utf-8"))
    items = {item["item_id"]: item for item in item_payload["items"]}
    rows = read_jsonl(source / "response_matrix.jsonl")
    cache = stage3.load_cache()
    lock = threading.Lock()
    calls: dict[tuple[str, str], dict[str, Any]] = {}
    to_call = [row for row in rows if row["domain"] in {"math", "factual"}]
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(stage3.call_verifier, items[row["item_id"]], row, cache, lock): row
            for row in to_call
        }
        for done, future in enumerate(as_completed(future_map), start=1):
            row = future_map[future]
            calls[(row["item_id"], row["action"])] = future.result()
            if done % 25 == 0:
                print(f"completed_fresh_verifier_calls={done}/{len(to_call)} elapsed_seconds={int(time.monotonic() - start)}")

    verified_rows = []
    raw_calls = []
    for row in rows:
        call = calls.get((row["item_id"], row["action"]))
        score, verdict, verifier_confidence = stage3.verifier_score(row, call)
        verified_action = f"{row['action']}_verified"
        verified_rows.append(
            {
                **row,
                "base_action": row["action"],
                "action": verified_action,
                "cost": stage3.VERIFIED_ACTIONS[verified_action],
                "base_confidence": row["confidence"],
                "verifier_score": round(score, 6),
                "verifier_verdict": verdict,
                "verifier_confidence": round(verifier_confidence, 6),
                "score": round(score, 6),
            }
        )
        if call is not None:
            raw_calls.append(call)

    write_jsonl(out / "fresh_verified_response_matrix.jsonl", verified_rows)
    write_jsonl(out / "fresh_verifier_call_manifest.jsonl", raw_calls)
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for call in raw_calls:
        for key in usage:
            value = (call.get("usage") or {}).get(key)
            if isinstance(value, int):
                usage[key] += value
    write_json(
        out / "fresh_verifier_summary.json",
        {
            "schema_version": "aira.ttc_fresh_verifier_summary.v1",
            "source_task": str(source),
            "model": args.verifier_model,
            "verified_row_count": len(verified_rows),
            "verifier_call_count": len(raw_calls),
            "cache_hit_count": sum(1 for call in raw_calls if call.get("cache_hit")),
            "usage_totals": usage,
            "elapsed_seconds": round(time.monotonic() - start, 3),
        },
    )


def group_items(rows: list[dict[str, Any]], split: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["split"] == split:
            grouped.setdefault(row["item_id"], []).append(row)
    return grouped


def as_float(value: Any, default: float = 0.0) -> float:
    return search_loop.as_float(value, default)


def score_row(row: dict[str, Any], model_spec: dict[str, Any], cost_penalty: float) -> float:
    proxy = {
        "domain": row["domain"],
        "action": row["action"],
        "verifier_score": row.get("verifier_score", row.get("score", 0.0)),
        "confidence": row.get("confidence", row.get("base_confidence", 0.0)),
        "agreement": row.get("agreement", 0.0),
        "candidate_count": row.get("candidate_count", 1.0),
        "cost": row.get("cost", 0.0),
        "verifier_tokens": row.get("verifier_total_tokens", 0.0),
    }
    return search_loop.torch_score(proxy, model_spec) - cost_penalty * as_float(row.get("cost"))


def best_action_for_item(item_rows: list[dict[str, Any]], model_spec: dict[str, Any], cost_penalty: float, cheap_delta: float) -> dict[str, Any]:
    choices = []
    for row in item_rows:
        score = score_row(row, model_spec, cost_penalty)
        choices.append((score, as_float(row.get("cost")), row))
    choices.sort(key=lambda item: (-item[0], item[1]))
    best_score = choices[0][0]
    feasible = [item for item in choices if item[0] >= best_score - cheap_delta]
    feasible.sort(key=lambda item: (item[1], -item[0]))
    score, _, row = feasible[0]
    return {**row, "policy_score": score}


def metric(decisions: list[dict[str, Any]], total_items: int) -> dict[str, Any]:
    accepted = [row for row in decisions if row.get("accepted")]
    errors = sum(1 for row in accepted if not row.get("correct"))
    accepted_count = len(accepted)
    total_cost = sum(as_float(row.get("cost")) for row in accepted)
    return {
        "n_items": total_items,
        "accepted_count": accepted_count,
        "accepted_errors": errors,
        "coverage": accepted_count / total_items if total_items else 0.0,
        "risk": errors / accepted_count if accepted_count else math.nan,
        "cost": total_cost / total_items if total_items else 0.0,
        "total_cost": total_cost,
    }


def evaluate_locked_policy(rows: list[dict[str, Any]], frozen: dict[str, Any], split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped = group_items(rows, split)
    model_spec = frozen["model_spec"]
    cost_penalty = float(frozen["cost_penalty"])
    cheap_delta = float(frozen["cheap_delta"])
    items = []
    for item_id, item_rows in grouped.items():
        chosen = best_action_for_item(item_rows, model_spec, cost_penalty, cheap_delta)
        items.append(chosen)
    items.sort(key=lambda row: (-as_float(row.get("policy_score")), as_float(row.get("cost"))))
    accept_count = min(len(items), max(0, math.ceil(float(frozen["target_coverage"]) * len(items) - 1e-9)))
    accepted_ids = {row["item_id"] for row in items[:accept_count]}
    decisions = []
    for row in items:
        accepted = row["item_id"] in accepted_ids
        decisions.append(
            {
                "policy": "locked_cost_compressed_policy",
                "split": split,
                "item_id": row["item_id"],
                "domain": row["domain"],
                "action": row["action"] if accepted else "abstain",
                "accepted": accepted,
                "correct": bool(row["correct"]) if accepted else "",
                "cost": as_float(row.get("cost")) if accepted else 0.0,
                "policy_score": row.get("policy_score"),
                "verifier_score": row.get("verifier_score"),
            }
        )
    return decisions, metric(decisions, len(grouped))


def tune_threshold(rows: list[dict[str, Any]], split: str, score_field: str, target_coverage: float) -> float:
    grouped = group_items(rows, split)
    scores = sorted({as_float(row.get(score_field)) for item_rows in grouped.values() for row in item_rows})
    thresholds = [max(scores) + 1e-6, *scores, min(scores) - 1e-6] if scores else [1.1]
    best_threshold = thresholds[0]
    best_key = (math.inf, math.inf, math.inf)
    for threshold in thresholds:
        decisions = []
        for item_id, item_rows in grouped.items():
            feasible = [row for row in item_rows if as_float(row.get(score_field)) >= threshold]
            if feasible:
                feasible.sort(key=lambda row: (-as_float(row.get(score_field)), as_float(row.get("cost"))))
                row = feasible[0]
                decisions.append({"accepted": True, "correct": bool(row["correct"]), "cost": as_float(row.get("cost"))})
            else:
                decisions.append({"accepted": False, "correct": "", "cost": 0.0})
        m = metric(decisions, len(grouped))
        risk = 1.0 if math.isnan(m["risk"]) else m["risk"]
        key = (abs(m["coverage"] - target_coverage), risk, m["cost"])
        if key < best_key:
            best_key = key
            best_threshold = threshold
    return best_threshold


def evaluate_threshold_baseline(rows: list[dict[str, Any]], split: str, score_field: str, threshold: float, policy_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped = group_items(rows, split)
    decisions = []
    for item_id, item_rows in grouped.items():
        feasible = [row for row in item_rows if as_float(row.get(score_field)) >= threshold]
        if feasible:
            feasible.sort(key=lambda row: (-as_float(row.get(score_field)), as_float(row.get("cost"))))
            row = feasible[0]
            decisions.append(
                {
                    "policy": policy_name,
                    "split": split,
                    "item_id": item_id,
                    "domain": row["domain"],
                    "action": row["action"],
                    "accepted": True,
                    "correct": bool(row["correct"]),
                    "cost": as_float(row.get("cost")),
                    "threshold": threshold,
                    score_field: row.get(score_field),
                }
            )
        else:
            domain = item_rows[0]["domain"]
            decisions.append(
                {
                    "policy": policy_name,
                    "split": split,
                    "item_id": item_id,
                    "domain": domain,
                    "action": "abstain",
                    "accepted": False,
                    "correct": "",
                    "cost": 0.0,
                    "threshold": threshold,
                }
            )
    return decisions, metric(decisions, len(grouped))


def evaluate(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((dep_dir("lock_policy") / "frozen_policy.json").read_text(encoding="utf-8"))
    rows = read_jsonl(dep_dir("verify_fresh_holdout") / "fresh_verified_response_matrix.jsonl")

    target_coverage = float(frozen["target_coverage"])
    locked_decisions, locked_metrics = evaluate_locked_policy(rows, frozen, "holdout")
    verifier_threshold = tune_threshold(rows, "tune", "verifier_score", target_coverage)
    confidence_threshold = tune_threshold(rows, "tune", "confidence", target_coverage)
    verifier_decisions, verifier_metrics = evaluate_threshold_baseline(
        rows, "holdout", "verifier_score", verifier_threshold, "fresh_matched_verifier_threshold"
    )
    confidence_decisions, confidence_metrics = evaluate_threshold_baseline(
        rows, "holdout", "confidence", confidence_threshold, "fresh_matched_confidence_threshold"
    )
    all_decisions = [*locked_decisions, *verifier_decisions, *confidence_decisions]
    metrics_rows = [
        {"policy": "locked_cost_compressed_policy", **locked_metrics},
        {"policy": "fresh_matched_verifier_threshold", **verifier_metrics, "threshold": verifier_threshold},
        {"policy": "fresh_matched_confidence_threshold", **confidence_metrics, "threshold": confidence_threshold},
    ]
    write_csv(out / "fresh_locked_holdout_metrics.csv", metrics_rows)
    write_csv(out / "fresh_locked_holdout_decisions.csv", all_decisions)

    primary = verifier_metrics
    locked_risk = 1.0 if math.isnan(locked_metrics["risk"]) else locked_metrics["risk"]
    baseline_risk = 1.0 if math.isnan(primary["risk"]) else primary["risk"]
    gate = {
        "coverage_gap_vs_fresh_verifier_baseline": locked_metrics["coverage"] - primary["coverage"],
        "risk_delta_vs_fresh_verifier_baseline": locked_risk - baseline_risk,
        "cost_delta_vs_fresh_verifier_baseline": locked_metrics["cost"] - primary["cost"],
        "pareto_dominates_fresh_verifier_baseline": (
            locked_metrics["coverage"] >= primary["coverage"] - 1e-9
            and locked_risk <= baseline_risk + 1e-9
            and locked_metrics["cost"] <= primary["cost"] + 1e-9
        ),
        "coverage_gap_vs_historical_baseline_target": locked_metrics["coverage"] - float(frozen["target_baseline"]["coverage"]),
        "risk_delta_vs_historical_baseline_target": locked_risk - float(frozen["target_baseline"]["risk"]),
        "cost_delta_vs_historical_baseline_target": locked_metrics["cost"] - float(frozen["target_baseline"]["cost"]),
    }
    status = "fresh_confirmed_pareto" if gate["pareto_dominates_fresh_verifier_baseline"] else "fresh_not_confirmed"
    claim_gate = {
        "schema_version": "aira.ttc_fresh_locked_holdout_claim_gate.v1",
        "status": status,
        "locked_policy": frozen["selected_policy"],
        "fresh_holdout": {
            "seed": FRESH_SEED,
            "n_items": locked_metrics["n_items"],
            "generated_at": now_iso(),
        },
        "locked_policy_metrics": locked_metrics,
        "fresh_verifier_baseline_metrics": primary,
        "fresh_confidence_baseline_metrics": confidence_metrics,
        "thresholds": {
            "fresh_verifier_threshold": verifier_threshold,
            "fresh_confidence_threshold": confidence_threshold,
        },
        "gate": gate,
        "interpretation": (
            "The frozen cost-compressed policy Pareto-dominated the fresh matched verifier-threshold baseline on locked holdout."
            if status == "fresh_confirmed_pareto"
            else "The frozen cost-compressed policy did not Pareto-dominate the fresh matched verifier-threshold baseline on locked holdout."
        ),
        "next_required_work": [
            "If fresh_confirmed_pareto, repeat at larger unique-item scale before manuscript claim finalization.",
            "If fresh_not_confirmed, inspect domain-level failures and collect richer verifier signals before drafting.",
        ],
    }
    write_json(out / "fresh_claim_gate.json", claim_gate)
    report = [
        "# Fresh Locked-Holdout Evaluation",
        "",
        f"Status: {status}",
        f"Locked policy: `{frozen['selected_policy']}`",
        "",
        "## Locked Policy Metrics",
        json.dumps(locked_metrics, indent=2, sort_keys=True),
        "",
        "## Fresh Matched Verifier Baseline",
        json.dumps(primary, indent=2, sort_keys=True),
        "",
        "## Gate",
        json.dumps(gate, indent=2, sort_keys=True),
        "",
    ]
    (out / "fresh_locked_holdout_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["lock_policy", "materialize", "verify", "evaluate"]:
        p = sub.add_parser(name)
        p.add_argument("--out", default=".")
    p = sub.choices["materialize"]
    p.add_argument("--seed", type=int, default=FRESH_SEED)
    p.add_argument("--math-items", type=int, default=48)
    p.add_argument("--factual-items", type=int, default=48)
    p.add_argument("--code-items", type=int, default=24)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--workers", type=int, default=6)
    p = sub.choices["verify"]
    p.add_argument("--verifier-model", default=DEFAULT_VERIFIER_MODEL)
    p.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "lock_policy":
        lock_policy(args)
    elif args.command == "materialize":
        materialize(args)
    elif args.command == "verify":
        verify(args)
    elif args.command == "evaluate":
        evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
