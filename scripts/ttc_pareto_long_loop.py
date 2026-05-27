#!/usr/bin/env python3
"""Long-running Pareto confirmation loop for verifier-assisted TTC policies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ttc_fresh_locked_holdout as fresh_eval
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
BASE_SEED = 2026052200
DEFAULT_MODEL = os.environ.get("ARA_TTC_MODEL", "gpt-4.1-mini")
DEFAULT_VERIFIER_MODEL = os.environ.get("ARA_TTC_VERIFIER_MODEL", "gpt-4.1-mini")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    if not keys:
        keys = ["status"]
        rows = [{"status": "empty"}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


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


def previous_state_path(round_index: int) -> Path:
    if round_index <= 1:
        return dep_dir("init_loop") / "loop_state.json"
    return dep_dir(f"round_{round_index - 1:02d}") / "loop_state.json"


def item_fingerprint(item: dict[str, Any]) -> str:
    return sha256_text(
        json.dumps(
            {
                "domain": item.get("domain"),
                "question": item.get("question"),
                "gold": item.get("gold"),
                "choices": item.get("choices"),
                "public_tests": item.get("public_tests"),
            },
            sort_keys=True,
            ensure_ascii=True,
        )
    )


def prior_fingerprints() -> set[str]:
    paths = [
        PRIOR_STAGE2_BUNDLE / "artifacts/tasks/materialize_real_outputs/dataset_items.json",
        PRIOR_POLICY_BUNDLE / "artifacts/tasks/materialize_real_outputs/dataset_items.json",
    ]
    fingerprints: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        payload = read_json(path)
        fingerprints.update(item_fingerprint(item) for item in payload.get("items", []))
    return fingerprints


def metric(decisions: list[dict[str, Any]], total_items: int) -> dict[str, Any]:
    accepted = [row for row in decisions if row.get("accepted")]
    accepted_count = len(accepted)
    errors = sum(1 for row in accepted if int(row.get("correct", 0)) == 0)
    total_cost = sum(search_loop.as_float(row.get("cost")) for row in accepted)
    return {
        "n_items": total_items,
        "accepted_count": accepted_count,
        "accepted_errors": errors,
        "coverage": accepted_count / total_items if total_items else 0.0,
        "risk": errors / accepted_count if accepted_count else math.nan,
        "cost": total_cost / total_items if total_items else 0.0,
        "total_cost": total_cost,
    }


def normalize_risk(value: Any) -> float:
    result = search_loop.as_float(value, math.nan)
    return 1.0 if math.isnan(result) else result


def old_sources() -> tuple[list[dict[str, str]], dict[str, Any], dict[str, float]]:
    rows = search_loop.read_csv(
        PRIOR_POLICY_BUNDLE / "artifacts/tasks/prepare_sources/canonical_policy_actions.csv"
    )
    model = read_json(PRIOR_POLICY_BUNDLE / "artifacts/tasks/search_policies/torch_meta_model.json")
    best = read_json(PRIOR_POLICY_BUNDLE / "artifacts/tasks/search_policies/best_policy_spec.json")
    baseline = {
        "coverage": float(best["baseline"]["coverage"]),
        "risk": float(best["baseline"]["risk"]),
        "cost": float(best["baseline"]["cost"]),
    }
    return rows, model, baseline


def candidate_grid() -> list[dict[str, Any]]:
    penalties = [
        0.0,
        0.005,
        0.01,
        0.015,
        0.02,
        0.025,
        0.03,
        0.035,
        0.04,
        0.05,
        0.075,
        0.1,
        0.15,
        0.2,
        0.3,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
    ]
    cheap_deltas = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1]
    candidates = []
    for penalty in penalties:
        for cheap_delta in cheap_deltas:
            name = f"gpu_torch_matched_coverage_p{penalty:.4f}_d{cheap_delta:.4f}".replace(".", "p")
            candidates.append(
                {
                    "policy": name,
                    "family": "gpu_torch_matched_coverage",
                    "cost_penalty": penalty,
                    "cheap_delta": cheap_delta,
                }
            )
    return candidates


def evaluate_old_candidate(
    rows: list[dict[str, str]],
    model: dict[str, Any],
    baseline: dict[str, float],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    decisions, specs = search_loop.evaluate_tune_budgeted_cost_compression(
        rows,
        model,
        baseline_coverage=baseline["coverage"],
        baseline_risk=baseline["risk"],
        baseline_cost=baseline["cost"],
        cost_penalty=float(candidate["cost_penalty"]),
        cheap_delta=float(candidate["cheap_delta"]),
    )
    m = search_loop.metrics(decisions)
    risk = normalize_risk(m["risk"])
    gate = {
        "old_coverage_gap_vs_baseline": float(m["coverage"]) - baseline["coverage"],
        "old_risk_delta_vs_baseline": risk - baseline["risk"],
        "old_cost_delta_vs_baseline": float(m["cost"]) - baseline["cost"],
        "old_pareto_dominates_baseline": (
            float(m["coverage"]) >= baseline["coverage"] - 1e-9
            and risk <= baseline["risk"] + 1e-9
            and float(m["cost"]) <= baseline["cost"] + 1e-9
        ),
    }
    return {
        **candidate,
        **m,
        **gate,
        "old_baseline_coverage": baseline["coverage"],
        "old_baseline_risk": baseline["risk"],
        "old_baseline_cost": baseline["cost"],
        "old_policy_spec": specs[0] if specs else {},
    }


def build_candidate_pool() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float], dict[str, Any]]:
    rows, model, baseline = old_sources()
    old_rows = [evaluate_old_candidate(rows, model, baseline, candidate) for candidate in candidate_grid()]
    pareto = [row for row in old_rows if row["old_pareto_dominates_baseline"]]
    pareto.sort(
        key=lambda row: (
            float(row["cost"]),
            normalize_risk(row["risk"]),
            -float(row["coverage"]),
            float(row["cost_penalty"]),
            float(row["cheap_delta"]),
        )
    )
    pool = []
    for row in pareto:
        pool.append(
            {
                "policy": row["policy"],
                "family": row["family"],
                "cost_penalty": row["cost_penalty"],
                "cheap_delta": row["cheap_delta"],
                "old_metrics": {
                    "n_items": row["n_items"],
                    "accepted_count": row["accepted_count"],
                    "accepted_errors": row["accepted_errors"],
                    "coverage": row["coverage"],
                    "risk": row["risk"],
                    "cost": row["cost"],
                },
                "old_gate": {
                    "coverage_gap_vs_baseline": row["old_coverage_gap_vs_baseline"],
                    "risk_delta_vs_baseline": row["old_risk_delta_vs_baseline"],
                    "cost_delta_vs_baseline": row["old_cost_delta_vs_baseline"],
                    "pareto_dominates_baseline": row["old_pareto_dominates_baseline"],
                },
                "stats": {
                    "fresh_attempts": 0,
                    "fresh_successes": 0,
                    "fresh_failures": 0,
                    "consecutive_fresh_successes": 0,
                },
            }
        )
    return pool, old_rows, baseline, model


def split_for(index: int, total: int) -> str:
    calibration_cut = max(1, int(round(total * 0.50)))
    tune_cut = max(calibration_cut + 1, int(round(total * 0.75)))
    if index < calibration_cut:
        return "calibration"
    if index < tune_cut:
        return "tune"
    return "holdout"


def load_unique_items(
    *,
    seed: int,
    math_items: int,
    factual_items: int,
    code_items: int,
    excluded: set[str],
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    stage2 = load_module(EXPERIMENTS_DIR / "run_real_benchmark_stage2.py", f"ara_ttc_stage2_items_{seed}")
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []

    def add_math() -> None:
        rows = list(load_dataset("gsm8k", "main", split="test"))
        rng.shuffle(rows)
        selected = 0
        for row in rows:
            gold = stage2.extract_number(row["answer"])
            item = {
                "item_id": f"gsm8k-r{seed}-{selected:03d}",
                "domain": "math",
                "split": split_for(selected, math_items),
                "question": row["question"],
                "gold": gold,
                "source_dataset": "gsm8k/main/test",
            }
            if item_fingerprint(item) in excluded:
                continue
            items.append(item)
            selected += 1
            if selected >= math_items:
                return
        raise RuntimeError("Not enough non-overlapping GSM8K items for fresh round.")

    def add_factual() -> None:
        rows = list(load_dataset("truthful_qa", "multiple_choice", split="validation"))
        rng.shuffle(rows)
        selected = 0
        for row in rows:
            targets = row["mc1_targets"]
            labels = list(targets["labels"])
            correct_index = labels.index(1)
            item = {
                "item_id": f"truthfulqa-r{seed}-{selected:03d}",
                "domain": "factual",
                "split": split_for(selected, factual_items),
                "question": row["question"],
                "choices": list(targets["choices"]),
                "gold": chr(ord("A") + correct_index),
                "source_dataset": "truthful_qa/multiple_choice/validation",
            }
            if item_fingerprint(item) in excluded:
                continue
            items.append(item)
            selected += 1
            if selected >= factual_items:
                return
        raise RuntimeError("Not enough non-overlapping TruthfulQA items for fresh round.")

    def add_code() -> None:
        rows = list(load_dataset("mbpp", split="test"))
        rng.shuffle(rows)
        selected = 0
        for row in rows:
            challenge_tests = list(row.get("challenge_test_list") or []) or list(row["test_list"])
            item = {
                "item_id": f"mbpp-r{seed}-{selected:03d}",
                "domain": "code",
                "split": split_for(selected, code_items),
                "question": row["text"],
                "public_tests": list(row["test_list"]),
                "challenge_tests": challenge_tests,
                "test_setup_code": row.get("test_setup_code") or "",
                "source_dataset": "mbpp/test",
            }
            if item_fingerprint(item) in excluded:
                continue
            items.append(item)
            selected += 1
            if selected >= code_items:
                return
        raise RuntimeError("Not enough non-overlapping MBPP items for fresh round.")

    add_math()
    add_factual()
    add_code()
    return items


def materialize_fresh_round(
    out: Path,
    *,
    seed: int,
    math_items: int,
    factual_items: int,
    code_items: int,
    workers: int,
    model: str,
    excluded: set[str],
) -> dict[str, Any]:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    items = load_unique_items(
        seed=seed,
        math_items=math_items,
        factual_items=factual_items,
        code_items=code_items,
        excluded=excluded,
    )
    fresh_fingerprints = [item_fingerprint(item) for item in items]
    overlap = sorted(set(fresh_fingerprints) & excluded)
    stage2 = load_module(EXPERIMENTS_DIR / "run_real_benchmark_stage2.py", f"ara_ttc_stage2_round_{seed}")
    stage2.SEED = seed
    stage2.MATH_ITEMS = math_items
    stage2.FACTUAL_ITEMS = factual_items
    stage2.CODE_ITEMS = code_items
    stage2.MODEL_ID = model
    stage2.MAX_WORKERS = workers
    stage2.CACHE_PATH = out / "fresh_stage2_openai_cache.json"
    stage2.load_items = lambda: items

    cwd = Path.cwd()
    os.chdir(out)
    try:
        stage2.materialize(argparse.Namespace())
    finally:
        os.chdir(cwd)

    audit = {
        "schema_version": "aira.ttc_long_loop_overlap_audit.v1",
        "seed": seed,
        "fresh_item_count": len(items),
        "excluded_fingerprint_count": len(excluded),
        "overlap_count": len(overlap),
        "overlap_fingerprints": overlap[:25],
        "zero_overlap": len(overlap) == 0,
        "locked_holdout_count": sum(1 for item in items if item.get("split") == "holdout"),
        "domain_counts": {
            domain: sum(1 for item in items if item.get("domain") == domain)
            for domain in sorted({item.get("domain") for item in items})
        },
        "fresh_fingerprints": fresh_fingerprints,
    }
    write_json(out / "fresh_overlap_audit.json", audit)
    return audit


def verify_fresh_round(out: Path, *, verifier_model: str, workers: int) -> dict[str, Any]:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    stage3 = load_module(EXPERIMENTS_DIR / "run_verifier_stage3.py", f"ara_ttc_stage3_round_{int(time.time())}")
    stage3.MODEL_ID = verifier_model
    stage3.MAX_WORKERS = workers
    stage3.CACHE_PATH = out / "fresh_stage3_verifier_cache.json"

    item_payload = read_json(out / "dataset_items.json")
    items = {item["item_id"]: item for item in item_payload["items"]}
    rows = read_jsonl(out / "response_matrix.jsonl")
    cache = stage3.load_cache()
    lock = threading.Lock()
    calls: dict[tuple[str, str], dict[str, Any]] = {}
    to_call = [row for row in rows if row["domain"] in {"math", "factual"}]
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(stage3.call_verifier, items[row["item_id"]], row, cache, lock): row
            for row in to_call
        }
        for done, future in enumerate(as_completed(future_map), start=1):
            row = future_map[future]
            calls[(row["item_id"], row["action"])] = future.result()
            if done % 25 == 0:
                print(f"completed_long_loop_verifier_calls={done}/{len(to_call)} elapsed_seconds={int(time.monotonic() - start)}")

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
    summary = {
        "schema_version": "aira.ttc_long_loop_verifier_summary.v1",
        "model": verifier_model,
        "verified_row_count": len(verified_rows),
        "verifier_call_count": len(raw_calls),
        "cache_hit_count": sum(1 for call in raw_calls if call.get("cache_hit")),
        "usage_totals": usage,
        "elapsed_seconds": round(time.monotonic() - start, 3),
    }
    write_json(out / "fresh_verifier_summary.json", summary)
    return summary


def evaluate_matched_candidate(
    rows: list[dict[str, Any]],
    model: dict[str, Any],
    candidate: dict[str, Any],
    baseline_accepted_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped = fresh_eval.group_items(rows, "holdout")
    ranked = []
    for item_id, item_rows in grouped.items():
        chosen = fresh_eval.best_action_for_item(
            item_rows,
            model,
            float(candidate["cost_penalty"]),
            float(candidate["cheap_delta"]),
        )
        ranked.append(chosen)
    ranked.sort(key=lambda row: (-fresh_eval.as_float(row.get("policy_score")), fresh_eval.as_float(row.get("cost"))))
    accepted_ids = {row["item_id"] for row in ranked[:baseline_accepted_count]}
    decisions = []
    for row in ranked:
        accepted = row["item_id"] in accepted_ids
        decisions.append(
            {
                "policy": candidate["policy"],
                "split": "holdout",
                "item_id": row["item_id"],
                "domain": row["domain"],
                "action": row["action"] if accepted else "abstain",
                "accepted": accepted,
                "correct": bool(row["correct"]) if accepted else "",
                "cost": fresh_eval.as_float(row.get("cost")) if accepted else 0.0,
                "policy_score": row.get("policy_score"),
                "verifier_score": row.get("verifier_score"),
            }
        )
    return decisions, metric(decisions, len(grouped))


def evaluate_fresh_candidates(
    out: Path,
    *,
    state: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    out = out.resolve()
    rows = read_jsonl(out / "fresh_verified_response_matrix.jsonl")
    model = state["model_spec"]
    old_baseline = state["old_baseline"]
    target_coverage = float(old_baseline["coverage"])

    verifier_threshold = fresh_eval.tune_threshold(rows, "tune", "verifier_score", target_coverage)
    confidence_threshold = fresh_eval.tune_threshold(rows, "tune", "confidence", target_coverage)
    verifier_decisions, verifier_metrics = fresh_eval.evaluate_threshold_baseline(
        rows,
        "holdout",
        "verifier_score",
        verifier_threshold,
        "fresh_matched_verifier_threshold",
    )
    confidence_decisions, confidence_metrics = fresh_eval.evaluate_threshold_baseline(
        rows,
        "holdout",
        "confidence",
        confidence_threshold,
        "fresh_matched_confidence_threshold",
    )
    baseline_accepted_count = int(verifier_metrics["accepted_count"])
    candidate_rows = []
    decision_rows = [*verifier_decisions, *confidence_decisions]
    successes = []
    for candidate in state["candidate_pool"]:
        decisions, candidate_metrics = evaluate_matched_candidate(rows, model, candidate, baseline_accepted_count)
        decision_rows.extend(decisions)
        candidate_risk = normalize_risk(candidate_metrics["risk"])
        baseline_risk = normalize_risk(verifier_metrics["risk"])
        gate = {
            "fresh_coverage_gap_vs_baseline": float(candidate_metrics["coverage"]) - float(verifier_metrics["coverage"]),
            "fresh_risk_delta_vs_baseline": candidate_risk - baseline_risk,
            "fresh_cost_delta_vs_baseline": float(candidate_metrics["cost"]) - float(verifier_metrics["cost"]),
            "fresh_pareto_dominates_baseline": (
                audit["zero_overlap"]
                and float(candidate_metrics["coverage"]) >= float(verifier_metrics["coverage"]) - 1e-9
                and candidate_risk <= baseline_risk + 1e-9
                and float(candidate_metrics["cost"]) <= float(verifier_metrics["cost"]) + 1e-9
            ),
        }
        row = {
            "policy": candidate["policy"],
            "cost_penalty": candidate["cost_penalty"],
            "cheap_delta": candidate["cheap_delta"],
            **{f"candidate_{key}": value for key, value in candidate_metrics.items()},
            **{f"baseline_{key}": value for key, value in verifier_metrics.items()},
            **gate,
        }
        candidate_rows.append(row)
        if gate["fresh_pareto_dominates_baseline"]:
            successes.append(row)

    candidate_rows.sort(
        key=lambda row: (
            not bool(row["fresh_pareto_dominates_baseline"]),
            float(row["fresh_cost_delta_vs_baseline"]),
            float(row["fresh_risk_delta_vs_baseline"]),
        )
    )
    write_csv(out / "candidate_fresh_metrics.csv", candidate_rows)
    write_csv(out / "fresh_policy_decisions.csv", decision_rows)
    result = {
        "verifier_threshold": verifier_threshold,
        "confidence_threshold": confidence_threshold,
        "fresh_verifier_baseline_metrics": verifier_metrics,
        "fresh_confidence_baseline_metrics": confidence_metrics,
        "baseline_accepted_count": baseline_accepted_count,
        "candidate_count": len(candidate_rows),
        "fresh_success_count": len(successes),
        "fresh_successes": successes[:25],
    }
    write_json(out / "fresh_evaluation_summary.json", result)
    return result


def empty_round_outputs(out: Path, state: dict[str, Any], reason: str) -> None:
    write_json(out / "loop_state.json", state)
    write_json(out / "round_claim_gate.json", {"schema_version": "aira.ttc_long_loop_round_gate.v1", "status": "skipped", "reason": reason})
    write_json(out / "fresh_overlap_audit.json", {"schema_version": "aira.ttc_long_loop_overlap_audit.v1", "status": "skipped", "reason": reason, "zero_overlap": True, "overlap_count": 0})
    write_csv(out / "candidate_old_metrics.csv", [])
    write_csv(out / "candidate_fresh_metrics.csv", [])
    (out / "round_report.md").write_text(f"# TTC Pareto Long Loop Round\n\nStatus: skipped\n\nReason: {reason}\n", encoding="utf-8")


def init_loop(args: argparse.Namespace) -> None:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    pool, old_rows, baseline, model = build_candidate_pool()
    if not pool:
        raise SystemExit("No old-data Pareto candidates found; cannot start long confirmation loop.")
    fingerprints = sorted(prior_fingerprints())
    state = {
        "schema_version": "aira.ttc_pareto_long_loop_state.v1",
        "status": "running",
        "started_at": now_iso(),
        "started_at_epoch": time.time(),
        "time_budget_hours": args.time_budget_hours,
        "max_rounds": args.max_rounds,
        "min_fresh_confirmations": args.min_confirmations,
        "rounds_completed": 0,
        "old_baseline": baseline,
        "model_spec": model,
        "candidate_pool": pool,
        "excluded_fingerprints": fingerprints,
        "best_current_policy": pool[0]["policy"],
        "resource_budget": {
            "gpu_fraction": 0.8,
            "network_required": True,
            "external_datasets_required": True,
            "live_model_calls": True,
            "model": args.model,
            "verifier_model": args.verifier_model,
            "math_items_per_round": args.math_items,
            "factual_items_per_round": args.factual_items,
            "code_items_per_round": args.code_items,
        },
    }
    write_json(out / "loop_state.json", state)
    write_csv(out / "candidate_old_metrics.csv", old_rows)
    write_json(
        out / "candidate_pool.json",
        {
            "schema_version": "aira.ttc_pareto_candidate_pool.v1",
            "old_pareto_candidate_count": len(pool),
            "old_candidate_count": len(old_rows),
            "selected_for_fresh_confirmation": pool,
        },
    )
    (out / "init_report.md").write_text(
        "\n".join(
            [
                "# TTC Pareto Long Loop Initialization",
                "",
                f"Status: running",
                f"Old-data Pareto candidates: {len(pool)}",
                f"Fresh confirmation target: {args.min_confirmations} consecutive zero-overlap rounds",
                f"Max rounds: {args.max_rounds}",
                f"Time budget hours: {args.time_budget_hours}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def update_candidate_stats(state: dict[str, Any], success_policies: set[str]) -> None:
    for candidate in state["candidate_pool"]:
        stats = candidate.setdefault("stats", {})
        stats["fresh_attempts"] = int(stats.get("fresh_attempts", 0)) + 1
        if candidate["policy"] in success_policies:
            stats["fresh_successes"] = int(stats.get("fresh_successes", 0)) + 1
            stats["consecutive_fresh_successes"] = int(stats.get("consecutive_fresh_successes", 0)) + 1
        else:
            stats["fresh_failures"] = int(stats.get("fresh_failures", 0)) + 1
            stats["consecutive_fresh_successes"] = 0


def run_round(args: argparse.Namespace) -> None:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    state = read_json(previous_state_path(args.round_index))
    elapsed_hours = (time.time() - float(state.get("started_at_epoch", time.time()))) / 3600.0
    if state.get("status") == "confirmed_pareto":
        empty_round_outputs(out, state, "loop already confirmed")
        return
    if elapsed_hours >= float(state.get("time_budget_hours", args.time_budget_hours)):
        state["status"] = "budget_exhausted"
        empty_round_outputs(out, state, "time budget exhausted")
        return

    seed = BASE_SEED + args.round_index
    excluded = set(state.get("excluded_fingerprints", []))
    audit = materialize_fresh_round(
        out,
        seed=seed,
        math_items=args.math_items,
        factual_items=args.factual_items,
        code_items=args.code_items,
        workers=args.workers,
        model=args.model,
        excluded=excluded,
    )
    verifier_summary = verify_fresh_round(out, verifier_model=args.verifier_model, workers=args.verifier_workers)
    fresh_summary = evaluate_fresh_candidates(out, state=state, audit=audit)

    old_rows = []
    for candidate in state["candidate_pool"]:
        old = candidate["old_metrics"]
        old_rows.append(
            {
                "policy": candidate["policy"],
                "cost_penalty": candidate["cost_penalty"],
                "cheap_delta": candidate["cheap_delta"],
                **old,
                **candidate["old_gate"],
            }
        )
    write_csv(out / "candidate_old_metrics.csv", old_rows)

    success_policies = {row["policy"] for row in fresh_summary["fresh_successes"]}
    update_candidate_stats(state, success_policies)
    state["rounds_completed"] = int(state.get("rounds_completed", 0)) + 1
    state["excluded_fingerprints"] = sorted(excluded | set(audit.get("fresh_fingerprints", [])))
    state.setdefault("round_history", []).append(
        {
            "round_index": args.round_index,
            "seed": seed,
            "status": "fresh_pareto_found" if success_policies else "fresh_pareto_not_found",
            "zero_overlap": audit["zero_overlap"],
            "overlap_count": audit["overlap_count"],
            "fresh_success_policies": sorted(success_policies),
            "fresh_verifier_baseline_metrics": fresh_summary["fresh_verifier_baseline_metrics"],
            "verifier_call_count": verifier_summary["verifier_call_count"],
        }
    )

    confirmed = [
        candidate
        for candidate in state["candidate_pool"]
        if int(candidate.get("stats", {}).get("consecutive_fresh_successes", 0))
        >= int(state["min_fresh_confirmations"])
    ]
    confirmed.sort(
        key=lambda candidate: (
            -int(candidate["stats"]["consecutive_fresh_successes"]),
            float(candidate["old_metrics"]["cost"]),
            normalize_risk(candidate["old_metrics"]["risk"]),
        )
    )
    if confirmed:
        state["status"] = "confirmed_pareto"
        state["confirmed_policy"] = confirmed[0]
    else:
        state["status"] = "running" if args.round_index < int(state["max_rounds"]) else "not_confirmed"

    best = sorted(
        state["candidate_pool"],
        key=lambda candidate: (
            -int(candidate.get("stats", {}).get("consecutive_fresh_successes", 0)),
            -int(candidate.get("stats", {}).get("fresh_successes", 0)),
            float(candidate["old_metrics"]["cost"]),
        ),
    )[0]
    state["best_current_policy"] = best["policy"]
    write_json(out / "loop_state.json", state)

    gate = {
        "schema_version": "aira.ttc_long_loop_round_gate.v1",
        "status": state["status"],
        "round_index": args.round_index,
        "seed": seed,
        "zero_overlap": audit["zero_overlap"],
        "fresh_success_count": fresh_summary["fresh_success_count"],
        "fresh_success_policies": sorted(success_policies),
        "best_current_policy": state["best_current_policy"],
        "confirmation_requirement": {
            "min_consecutive_fresh_successes": state["min_fresh_confirmations"],
            "old_data_pareto_required": True,
            "fresh_data_zero_overlap_required": True,
            "matched_coverage_pareto_gate": True,
        },
    }
    write_json(out / "round_claim_gate.json", gate)
    (out / "round_report.md").write_text(
        "\n".join(
            [
                "# TTC Pareto Long Loop Round",
                "",
                f"Status: {state['status']}",
                f"Round: {args.round_index}",
                f"Seed: {seed}",
                f"Zero overlap: {audit['zero_overlap']}",
                f"Fresh Pareto successes: {fresh_summary['fresh_success_count']}",
                f"Best current policy: `{state['best_current_policy']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def finalize(args: argparse.Namespace) -> None:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    deps = json.loads(os.environ.get("AIRA_DEP_DIRS", "{}"))
    round_ids = sorted(task_id for task_id in deps if task_id.startswith("round_"))
    state_path = Path(deps[round_ids[-1]]) / "loop_state.json" if round_ids else dep_dir("init_loop") / "loop_state.json"
    state = read_json(state_path)
    rows = []
    for candidate in state.get("candidate_pool", []):
        stats = candidate.get("stats", {})
        rows.append(
            {
                "policy": candidate["policy"],
                "cost_penalty": candidate["cost_penalty"],
                "cheap_delta": candidate["cheap_delta"],
                **{f"old_{key}": value for key, value in candidate["old_metrics"].items()},
                **{f"old_gate_{key}": value for key, value in candidate["old_gate"].items()},
                **stats,
            }
        )
    rows.sort(
        key=lambda row: (
            -int(row.get("consecutive_fresh_successes", 0)),
            -int(row.get("fresh_successes", 0)),
            float(row.get("old_cost", 999.0)),
        )
    )
    write_csv(out / "candidate_confirmation_ledger.csv", rows)
    gate = {
        "schema_version": "aira.ttc_pareto_long_loop_claim_gate.v1",
        "status": state.get("status"),
        "rounds_completed": state.get("rounds_completed", 0),
        "best_current_policy": state.get("best_current_policy"),
        "confirmed_policy": state.get("confirmed_policy"),
        "old_baseline": state.get("old_baseline"),
        "confirmation_requirement": {
            "old_data_pareto_required": True,
            "fresh_data_zero_overlap_required": True,
            "min_consecutive_fresh_successes": state.get("min_fresh_confirmations"),
            "matched_coverage_pareto_gate": True,
        },
        "interpretation": (
            "A single policy passed the old-data Pareto gate and the required consecutive zero-overlap fresh matched-coverage Pareto gates."
            if state.get("status") == "confirmed_pareto"
            else "The long loop did not yet confirm a single policy under the old-data and repeated zero-overlap fresh matched-coverage Pareto gates."
        ),
    }
    write_json(out / "long_loop_claim_gate.json", gate)
    write_json(out / "final_loop_state.json", state)
    (out / "pareto_long_loop_report.md").write_text(
        "\n".join(
            [
                "# TTC Pareto Long Loop Final Report",
                "",
                f"Status: {state.get('status')}",
                f"Rounds completed: {state.get('rounds_completed', 0)}",
                f"Best current policy: `{state.get('best_current_policy')}`",
                "",
                "## Interpretation",
                gate["interpretation"],
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_loop(args: argparse.Namespace) -> None:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    task_dirs: dict[str, str] = {}

    init_dir = out / "init"
    init_args = argparse.Namespace(
        out=str(init_dir),
        max_rounds=args.max_rounds,
        time_budget_hours=args.time_budget_hours,
        min_confirmations=args.min_confirmations,
        math_items=args.math_items,
        factual_items=args.factual_items,
        code_items=args.code_items,
        model=args.model,
        verifier_model=args.verifier_model,
    )
    init_loop(init_args)
    task_dirs["init_loop"] = str(init_dir)
    latest_state = read_json(init_dir / "loop_state.json")
    round_manifest = [{"task_id": "init_loop", "path": str(init_dir), "status": latest_state["status"]}]

    for round_index in range(1, args.max_rounds + 1):
        if latest_state.get("status") in {"confirmed_pareto", "budget_exhausted", "not_confirmed"}:
            break
        round_id = f"round_{round_index:02d}"
        round_dir = out / round_id
        deps = {"init_loop": str(init_dir)}
        if round_index > 1:
            deps[f"round_{round_index - 1:02d}"] = str(out / f"round_{round_index - 1:02d}")
        os.environ["AIRA_DEP_DIRS"] = json.dumps(deps, sort_keys=True)
        round_args = argparse.Namespace(
            out=str(round_dir),
            round_index=round_index,
            time_budget_hours=args.time_budget_hours,
            math_items=args.math_items,
            factual_items=args.factual_items,
            code_items=args.code_items,
            workers=args.workers,
            verifier_workers=args.verifier_workers,
            model=args.model,
            verifier_model=args.verifier_model,
        )
        run_round(round_args)
        task_dirs[round_id] = str(round_dir)
        latest_state = read_json(round_dir / "loop_state.json")
        round_manifest.append(
            {
                "task_id": round_id,
                "path": str(round_dir),
                "status": latest_state.get("status"),
                "best_current_policy": latest_state.get("best_current_policy"),
            }
        )

    final_dir = out / "final"
    deps = {"init_loop": str(init_dir)}
    for item in round_manifest:
        task_id = str(item["task_id"])
        if task_id.startswith("round_"):
            deps[task_id] = str(item["path"])
    os.environ["AIRA_DEP_DIRS"] = json.dumps(deps, sort_keys=True)
    finalize(argparse.Namespace(out=str(final_dir)))

    for name in [
        "long_loop_claim_gate.json",
        "candidate_confirmation_ledger.csv",
        "pareto_long_loop_report.md",
        "final_loop_state.json",
    ]:
        shutil.copy2(final_dir / name, out / name)
    write_json(
        out / "round_artifact_manifest.json",
        {
            "schema_version": "aira.ttc_pareto_long_loop_artifact_manifest.v1",
            "root": str(out),
            "rounds": round_manifest,
            "final_dir": str(final_dir),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--out", default=".")
    init.add_argument("--max-rounds", type=int, default=20)
    init.add_argument("--time-budget-hours", type=float, default=12.0)
    init.add_argument("--min-confirmations", type=int, default=3)
    init.add_argument("--math-items", type=int, default=48)
    init.add_argument("--factual-items", type=int, default=48)
    init.add_argument("--code-items", type=int, default=24)
    init.add_argument("--model", default=DEFAULT_MODEL)
    init.add_argument("--verifier-model", default=DEFAULT_VERIFIER_MODEL)

    run = sub.add_parser("run-round")
    run.add_argument("--out", default=".")
    run.add_argument("--round-index", type=int, required=True)
    run.add_argument("--time-budget-hours", type=float, default=12.0)
    run.add_argument("--math-items", type=int, default=48)
    run.add_argument("--factual-items", type=int, default=48)
    run.add_argument("--code-items", type=int, default=24)
    run.add_argument("--workers", type=int, default=6)
    run.add_argument("--verifier-workers", type=int, default=8)
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--verifier-model", default=DEFAULT_VERIFIER_MODEL)

    final = sub.add_parser("finalize")
    final.add_argument("--out", default=".")

    loop = sub.add_parser("run-loop")
    loop.add_argument("--out", default=".")
    loop.add_argument("--max-rounds", type=int, default=20)
    loop.add_argument("--time-budget-hours", type=float, default=12.0)
    loop.add_argument("--min-confirmations", type=int, default=3)
    loop.add_argument("--math-items", type=int, default=48)
    loop.add_argument("--factual-items", type=int, default=48)
    loop.add_argument("--code-items", type=int, default=24)
    loop.add_argument("--workers", type=int, default=6)
    loop.add_argument("--verifier-workers", type=int, default=8)
    loop.add_argument("--model", default=DEFAULT_MODEL)
    loop.add_argument("--verifier-model", default=DEFAULT_VERIFIER_MODEL)

    args = parser.parse_args()
    if args.command == "init":
        init_loop(args)
    elif args.command == "run-round":
        run_round(args)
    elif args.command == "finalize":
        finalize(args)
    elif args.command == "run-loop":
        run_loop(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
