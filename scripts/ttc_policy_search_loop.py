#!/usr/bin/env python3
"""AIRA policy-search loop for verifier-assisted selective TTC.

This script is intentionally self-contained so it can be called from an AIRA
production-open plan as an external command. It reuses the existing ARA/AIRA
evidence bundle as a seed, searches new selective policies against the strong
threshold baseline, and emits machine-readable artifacts for the result bundle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(
    "/home/biostar/work/projects/ara-manuscript-private/"
    "ara_manuscript_private/workspaces/conformal-ttc-risk-control-20260521"
)
LEDGER_PATH = PROJECT_ROOT / "exp/revrun_20260521_141121_r01_task_01/unified_evaluation_ledger_repeated_splits.csv"
BASELINE_PATH = PROJECT_ROOT / "exp/revrun_20260521_191840_r06_task_02/baseline_comparison.csv"
CLAIM_STATUS_PATH = PROJECT_ROOT / "exp/revrun_20260521_191840_r06_task_03/claim_status.json"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return default
    if text.lower() == "true":
        return 1.0
    if text.lower() == "false":
        return 0.0
    try:
        return float(text)
    except ValueError:
        return default


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def gpu_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "aira.gpu_resource_report.v1",
        "requested_max_memory_fraction": 0.8,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "torch_available": False,
        "cuda_available": False,
        "devices": [],
    }
    try:
        import torch

        report["torch_available"] = True
        report["torch_version"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                report["devices"].append(
                    {
                        "index": index,
                        "name": props.name,
                        "total_memory_bytes": int(props.total_memory),
                        "budget_memory_bytes": int(props.total_memory * 0.8),
                    }
                )
            try:
                torch.cuda.set_per_process_memory_fraction(0.8, device=0)
                report["memory_fraction_set"] = True
            except Exception as exc:  # pragma: no cover - hardware-dependent
                report["memory_fraction_set"] = False
                report["memory_fraction_error"] = repr(exc)
    except Exception as exc:
        report["torch_error"] = repr(exc)
    return report


def prepare(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_csv(LEDGER_PATH)
    canonical: list[dict[str, Any]] = []
    for row in rows:
        split_id = row.get("repeated_split_id") or row.get("repeated_split") or "repeat_00"
        action = row.get("verified_action") or row.get("base_action") or "unknown"
        cost = as_float(row.get("normalized_cost_total_verified_action"), as_float(row.get("normalized_cost_total_pipeline"), 1.0))
        canonical.append(
            {
                "repeat_id": split_id,
                "partition": row.get("current_split", ""),
                "domain": row.get("domain", ""),
                "item_id": row.get("item_id", ""),
                "action": action,
                "correct": int(as_bool(row.get("verified_correct", row.get("final_correct")))),
                "verifier_score": as_float(row.get("verifier_score")),
                "confidence": as_float(row.get("verified_confidence"), as_float(row.get("base_confidence"))),
                "agreement": as_float(row.get("verified_agreement"), as_float(row.get("base_agreement"))),
                "candidate_count": as_float(row.get("verified_candidate_count"), as_float(row.get("base_candidate_count"), 1.0)),
                "call_error_count": as_float(row.get("verified_call_error_count"), 0.0),
                "verifier_tokens": as_float(row.get("verifier_total_tokens"), 0.0),
                "cost": cost,
                "source_dataset": row.get("source_dataset", ""),
            }
        )
    write_csv(out / "canonical_policy_actions.csv", canonical)
    baseline_rows = read_csv(BASELINE_PATH) if BASELINE_PATH.exists() else []
    write_csv(out / "baseline_reference.csv", baseline_rows)
    inventory = {
        "schema_version": "aira.ttc_policy_source_inventory.v1",
        "source_project": str(PROJECT_ROOT),
        "input_files": [
            {"path": str(LEDGER_PATH), "sha256": sha256_file(LEDGER_PATH), "size_bytes": LEDGER_PATH.stat().st_size},
            {"path": str(BASELINE_PATH), "sha256": sha256_file(BASELINE_PATH), "size_bytes": BASELINE_PATH.stat().st_size},
            {"path": str(CLAIM_STATUS_PATH), "sha256": sha256_file(CLAIM_STATUS_PATH), "size_bytes": CLAIM_STATUS_PATH.stat().st_size},
        ],
        "row_count": len(canonical),
        "repeat_ids": sorted({row["repeat_id"] for row in canonical}),
        "partitions": sorted({row["partition"] for row in canonical}),
        "domains": sorted({row["domain"] for row in canonical}),
        "actions": sorted({row["action"] for row in canonical}),
        "baseline_claim_boundary": json.loads(CLAIM_STATUS_PATH.read_text(encoding="utf-8")),
    }
    write_json(out / "source_inventory.json", inventory)
    write_json(out / "gpu_resource_report.json", gpu_report())


def dep_dir(task_id: str) -> Path:
    dep_dirs = json.loads(os.environ.get("AIRA_DEP_DIRS", "{}"))
    if task_id not in dep_dirs:
        raise SystemExit(f"Missing dependency task dir for {task_id}")
    return Path(dep_dirs[task_id])


def load_actions_from_prepare() -> list[dict[str, str]]:
    return read_csv(dep_dir("prepare_sources") / "canonical_policy_actions.csv")


def rows_for(rows: list[dict[str, str]], repeat_id: str, partition: str) -> list[dict[str, str]]:
    return [row for row in rows if row["repeat_id"] == repeat_id and row["partition"] == partition]


def group_items(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["domain"], row["item_id"]), []).append(row)
    return grouped


def item_domain(item_key: tuple[str, str]) -> str:
    return item_key[0]


def compute_policy_score(row: dict[str, str], family: str, weights: dict[str, float] | None = None) -> float:
    weights = weights or {}
    verifier = as_float(row.get("verifier_score"))
    confidence = as_float(row.get("confidence"))
    agreement = as_float(row.get("agreement"))
    cost = as_float(row.get("cost"))
    candidate_count = as_float(row.get("candidate_count"), 1.0)
    token_cost = min(as_float(row.get("verifier_tokens")) / 500.0, 2.0)
    if family == "verifier_threshold":
        return verifier
    if family == "confidence_threshold":
        return confidence
    if family == "agreement_threshold":
        return agreement
    if family == "cost_aware_meta_score":
        return (
            weights.get("verifier", 0.55) * verifier
            + weights.get("confidence", 0.25) * confidence
            + weights.get("agreement", 0.15) * agreement
            + weights.get("candidate_count", 0.03) * min(candidate_count / 3.0, 1.0)
            - weights.get("cost", 0.08) * cost
            - weights.get("token_cost", 0.02) * token_cost
        )
    raise ValueError(f"unknown policy family: {family}")


def select_candidate(
    candidates: list[dict[str, str]],
    family: str,
    threshold: float,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    scored = []
    for row in candidates:
        score = compute_policy_score(row, family, weights)
        if score >= threshold:
            scored.append((score, as_float(row.get("cost")), row))
    if not scored:
        return {"accepted": False}
    scored.sort(key=lambda item: (-item[0], item[1]))
    score, _, row = scored[0]
    return {
        "accepted": True,
        "domain": row["domain"],
        "item_id": row["item_id"],
        "action": row["action"],
        "score": score,
        "correct": int(as_float(row.get("correct"))),
        "cost": as_float(row.get("cost")),
    }


def metrics(decisions: list[dict[str, Any]]) -> dict[str, float]:
    total = len(decisions)
    accepted = [row for row in decisions if row.get("accepted")]
    accepted_count = len(accepted)
    errors = sum(1 for row in accepted if int(row.get("correct", 0)) == 0)
    cost = sum(as_float(row.get("cost")) for row in accepted) / total if total else 0.0
    return {
        "n_items": total,
        "accepted_count": accepted_count,
        "accepted_errors": errors,
        "coverage": accepted_count / total if total else 0.0,
        "risk": errors / accepted_count if accepted_count else math.nan,
        "cost": cost,
    }


def threshold_grid(scores: list[float]) -> list[float]:
    if not scores:
        return [1.1]
    unique = sorted(set(round(score, 6) for score in scores))
    quantiles = []
    for q in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        index = min(len(unique) - 1, max(0, int(q * (len(unique) - 1))))
        quantiles.append(unique[index])
    return sorted(set(unique + quantiles + [min(unique) - 1e-6, max(unique) + 1e-6]))


def tune_threshold(
    train_rows: list[dict[str, str]],
    family: str,
    target_coverage: float,
    weights: dict[str, float] | None = None,
    domain: str | None = None,
) -> float:
    rows = [row for row in train_rows if domain is None or row["domain"] == domain]
    grouped = group_items(rows)
    scores = [compute_policy_score(row, family, weights) for row in rows]
    best_threshold = max(scores) + 1.0 if scores else 1.1
    best_key = (999.0, 999.0, 999.0)
    for threshold in threshold_grid(scores):
        decisions = [select_candidate(cands, family, threshold, weights) | {"domain": key[0], "item_id": key[1]} for key, cands in grouped.items()]
        m = metrics(decisions)
        risk = m["risk"]
        if math.isnan(risk):
            risk = 1.0
        coverage_penalty = max(0.0, target_coverage - m["coverage"]) * 2.0
        high_coverage_bonus = -0.1 * min(m["coverage"], target_coverage + 0.05)
        key = (risk + coverage_penalty + 0.04 * m["cost"] + high_coverage_bonus, abs(m["coverage"] - target_coverage), m["cost"])
        if key < best_key:
            best_key = key
            best_threshold = threshold
    return best_threshold


def evaluate_threshold_family(
    rows: list[dict[str, str]],
    family: str,
    target_by_domain: dict[str, float],
    weights: dict[str, float] | None = None,
    domain_specific: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repeats = sorted({row["repeat_id"] for row in rows})
    all_decisions: list[dict[str, Any]] = []
    specs: list[dict[str, Any]] = []
    for repeat_id in repeats:
        tune_rows = rows_for(rows, repeat_id, "tune")
        holdout_rows = rows_for(rows, repeat_id, "holdout")
        grouped = group_items(holdout_rows)
        thresholds: dict[str, float] = {}
        if domain_specific:
            for domain, target in target_by_domain.items():
                thresholds[domain] = tune_threshold(tune_rows, family, target, weights=weights, domain=domain)
        else:
            thresholds["overall"] = tune_threshold(tune_rows, family, target_by_domain.get("overall", 0.35), weights=weights)
        for key, cands in grouped.items():
            domain = item_domain(key)
            threshold = thresholds.get(domain, thresholds.get("overall", 1.1))
            decision = select_candidate(cands, family, threshold, weights)
            decision.update({"repeat_id": repeat_id, "domain": domain, "item_id": key[1], "policy": family})
            all_decisions.append(decision)
        specs.append({"repeat_id": repeat_id, "family": family, "thresholds": thresholds, "weights": weights or {}, "domain_specific": domain_specific})
    return all_decisions, specs


def one_hot(value: str, values: list[str]) -> list[float]:
    return [1.0 if value == item else 0.0 for item in values]


def train_torch_meta(rows: list[dict[str, str]], out: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"schema_version": "aira.torch_meta_training_report.v1", "used_torch": False}
    try:
        import torch
    except Exception as exc:
        report["error"] = repr(exc)
        write_json(out / "gpu_training_report.json", report)
        return report

    random_seed = 20260522
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    domains = sorted({row["domain"] for row in rows})
    actions = sorted({row["action"] for row in rows})
    train_rows = [row for row in rows if row["partition"] in {"calibration", "tune"}]
    if not train_rows:
        write_json(out / "gpu_training_report.json", report)
        return report

    def featurize(row: dict[str, str]) -> list[float]:
        return [
            as_float(row.get("verifier_score")),
            as_float(row.get("confidence")),
            as_float(row.get("agreement")),
            min(as_float(row.get("candidate_count"), 1.0) / 3.0, 1.0),
            as_float(row.get("cost")) / 6.0,
            min(as_float(row.get("verifier_tokens")) / 500.0, 2.0),
            *one_hot(row["domain"], domains),
            *one_hot(row["action"], actions),
        ]

    x = torch.tensor([featurize(row) for row in train_rows], dtype=torch.float32)
    y = torch.tensor([[as_float(row.get("correct"))] for row in train_rows], dtype=torch.float32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            torch.cuda.set_per_process_memory_fraction(0.8, device=0)
        except Exception:
            pass
    x = x.to(device)
    y = y.to(device)
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 1), torch.nn.Sigmoid()).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.08, weight_decay=0.01)
    loss_fn = torch.nn.BCELoss()
    losses = []
    for _ in range(300):
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        probs = model(x).detach().cpu().reshape(-1).tolist()
    report.update(
        {
            "used_torch": True,
            "device": str(device),
            "domains": domains,
            "actions": actions,
            "feature_count": int(x.shape[1]),
            "train_rows": len(train_rows),
            "random_seed": random_seed,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "positive_rate": float(sum(as_float(row.get("correct")) for row in train_rows) / len(train_rows)),
            "probability_mean": statistics.mean(probs),
        }
    )
    weights = model[0].weight.detach().cpu().reshape(-1).tolist()
    bias = float(model[0].bias.detach().cpu().reshape(-1)[0])
    write_json(out / "torch_meta_model.json", {"domains": domains, "actions": actions, "weights": weights, "bias": bias})
    write_json(out / "gpu_training_report.json", report)
    return report


def torch_score(row: dict[str, str], model_spec: dict[str, Any]) -> float:
    domains = list(model_spec["domains"])
    actions = list(model_spec["actions"])
    features = [
        as_float(row.get("verifier_score")),
        as_float(row.get("confidence")),
        as_float(row.get("agreement")),
        min(as_float(row.get("candidate_count"), 1.0) / 3.0, 1.0),
        as_float(row.get("cost")) / 6.0,
        min(as_float(row.get("verifier_tokens")) / 500.0, 2.0),
        *one_hot(row["domain"], domains),
        *one_hot(row["action"], actions),
    ]
    z = float(model_spec["bias"]) + sum(float(w) * x for w, x in zip(model_spec["weights"], features))
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def evaluate_torch_meta(rows: list[dict[str, str]], model_spec: dict[str, Any], target_by_domain: dict[str, float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repeats = sorted({row["repeat_id"] for row in rows})
    all_decisions: list[dict[str, Any]] = []
    specs: list[dict[str, Any]] = []

    def select(cands: list[dict[str, str]], threshold: float) -> dict[str, Any]:
        scored = []
        for row in cands:
            score = torch_score(row, model_spec) - 0.03 * as_float(row.get("cost"))
            if score >= threshold:
                scored.append((score, as_float(row.get("cost")), row))
        if not scored:
            return {"accepted": False}
        scored.sort(key=lambda item: (-item[0], item[1]))
        score, _, row = scored[0]
        return {"accepted": True, "action": row["action"], "score": score, "correct": int(as_float(row.get("correct"))), "cost": as_float(row.get("cost"))}

    for repeat_id in repeats:
        tune_rows = rows_for(rows, repeat_id, "tune")
        holdout_rows = rows_for(rows, repeat_id, "holdout")
        thresholds: dict[str, float] = {}
        for domain, target in target_by_domain.items():
            if domain == "overall":
                continue
            domain_rows = [row for row in tune_rows if row["domain"] == domain]
            grouped = group_items(domain_rows)
            scores = [torch_score(row, model_spec) - 0.03 * as_float(row.get("cost")) for row in domain_rows]
            best_t = max(scores) + 1 if scores else 2.0
            best_key = (999.0, 999.0)
            for threshold in threshold_grid(scores):
                decisions = [select(cands, threshold) for cands in grouped.values()]
                m = metrics(decisions)
                risk = 1.0 if math.isnan(m["risk"]) else m["risk"]
                key = (risk + max(0.0, target - m["coverage"]) * 2.0 + 0.04 * m["cost"], abs(m["coverage"] - target))
                if key < best_key:
                    best_key = key
                    best_t = threshold
            thresholds[domain] = best_t
        for key, cands in group_items(holdout_rows).items():
            domain = key[0]
            decision = select(cands, thresholds.get(domain, 2.0))
            decision.update({"repeat_id": repeat_id, "domain": domain, "item_id": key[1], "policy": "gpu_torch_meta_risk"})
            all_decisions.append(decision)
        specs.append({"repeat_id": repeat_id, "family": "gpu_torch_meta_risk", "thresholds": thresholds})
    return all_decisions, specs


def evaluate_torch_budgeted_rank(
    rows: list[dict[str, str]],
    model_spec: dict[str, Any],
    target_coverage: float,
    cost_penalty: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select a fixed global acceptance budget using the learned risk score.

    The earlier threshold policies tune each domain independently. That can be
    too conservative when one domain has little signal. This policy spends the
    acceptance budget globally and lets low-risk domains absorb the coverage.
    """

    repeats = sorted({row["repeat_id"] for row in rows})
    all_decisions: list[dict[str, Any]] = []
    specs: list[dict[str, Any]] = []
    for repeat_id in repeats:
        holdout_rows = rows_for(rows, repeat_id, "holdout")
        grouped = group_items(holdout_rows)
        item_scores: list[tuple[float, float, tuple[str, str], dict[str, str]]] = []
        for key, cands in grouped.items():
            scored = []
            for row in cands:
                score = torch_score(row, model_spec) - cost_penalty * as_float(row.get("cost"))
                scored.append((score, as_float(row.get("cost")), row))
            scored.sort(key=lambda item: (-item[0], item[1]))
            score, cost, row = scored[0]
            item_scores.append((score, cost, key, row))
        item_scores.sort(key=lambda item: (-item[0], item[1]))
        accept_count = min(len(item_scores), max(0, round(target_coverage * len(item_scores))))
        accepted_keys = {key for _, _, key, _ in item_scores[:accept_count]}
        cutoff_score = item_scores[accept_count - 1][0] if accept_count else None
        for score, _, key, row in item_scores:
            accepted = key in accepted_keys
            decision: dict[str, Any] = {
                "accepted": accepted,
                "repeat_id": repeat_id,
                "domain": key[0],
                "item_id": key[1],
                "policy": "gpu_torch_budgeted_rank",
            }
            if accepted:
                decision.update(
                    {
                        "action": row["action"],
                        "score": score,
                        "correct": int(as_float(row.get("correct"))),
                        "cost": as_float(row.get("cost")),
                    }
                )
            all_decisions.append(decision)
        specs.append(
            {
                "repeat_id": repeat_id,
                "family": "gpu_torch_budgeted_rank",
                "target_coverage": target_coverage,
                "cost_penalty": cost_penalty,
                "accepted_items": accept_count,
                "cutoff_score": cutoff_score,
            }
        )
    return all_decisions, specs


def ranked_torch_items(
    rows: list[dict[str, str]],
    repeat_id: str,
    partition: str,
    model_spec: dict[str, Any],
    cost_penalty: float,
    cheap_delta: float = 0.0,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for key, cands in group_items(rows_for(rows, repeat_id, partition)).items():
        choices = []
        for row in cands:
            base_score = torch_score(row, model_spec)
            score = base_score - cost_penalty * as_float(row.get("cost"))
            choices.append(
                {
                    "score": score,
                    "base_score": base_score,
                    "cost": as_float(row.get("cost")),
                    "correct": int(as_float(row.get("correct"))),
                    "domain": key[0],
                    "item_id": key[1],
                    "action": row["action"],
                }
            )
        choices.sort(key=lambda item: (-item["score"], item["cost"]))
        best_score = choices[0]["score"]
        feasible = [item for item in choices if item["score"] >= best_score - cheap_delta]
        feasible.sort(key=lambda item: (item["cost"], -item["score"]))
        items.append(feasible[0])
    items.sort(key=lambda item: (-item["score"], item["cost"]))
    return items


def _item_curve(items: list[dict[str, Any]]) -> list[tuple[int, int, float]]:
    curve = [(0, 0, 0.0)]
    errors = 0
    cost = 0.0
    for accepted_count, item in enumerate(items, start=1):
        errors += 1 - int(item["correct"])
        cost += as_float(item["cost"])
        curve.append((accepted_count, errors, cost))
    return curve


def _tune_budget_allocation(
    curves: dict[str, list[tuple[int, int, float]]],
    *,
    baseline_coverage: float,
    baseline_risk: float,
    baseline_cost: float,
) -> dict[str, Any]:
    total_items = sum(max(k for k, _, _ in curve) for curve in curves.values())
    target_accepts = math.ceil(baseline_coverage * total_items - 1e-9)
    states: dict[tuple[int, int], tuple[float, list[dict[str, Any]]]] = {(0, 0): (0.0, [])}
    for repeat_id, curve in curves.items():
        next_states: dict[tuple[int, int], tuple[float, list[dict[str, Any]]]] = {}
        for (accepted, errors), (cost, choice) in states.items():
            for k, repeat_errors, repeat_cost in curve:
                key = (accepted + k, errors + repeat_errors)
                value = cost + repeat_cost
                if key not in next_states or value < next_states[key][0]:
                    next_states[key] = (
                        value,
                        [
                            *choice,
                            {
                                "repeat_id": repeat_id,
                                "accepted_items": k,
                                "errors": repeat_errors,
                                "cost": repeat_cost,
                            },
                        ],
                    )
        states = next_states

    ranked = []
    for (accepted, errors), (cost, choice) in states.items():
        if accepted < target_accepts:
            continue
        coverage = accepted / total_items if total_items else 0.0
        risk = errors / accepted if accepted else math.inf
        cost_metric = cost / total_items if total_items else math.inf
        tune_pareto = risk <= baseline_risk + 1e-9 and cost_metric <= baseline_cost + 1e-9
        score = (
            (0.0 if tune_pareto else 1.0)
            + max(0.0, risk - baseline_risk) * 5.0
            + max(0.0, cost_metric - baseline_cost)
            + abs(coverage - baseline_coverage) * 0.1
            + risk * 0.02
            + cost_metric * 0.01
        )
        ranked.append(
            {
                "score": score,
                "accepted_items": accepted,
                "errors": errors,
                "coverage": coverage,
                "risk": risk,
                "cost": cost_metric,
                "allocation": choice,
                "tune_total_items": total_items,
                "target_accepts": target_accepts,
                "tune_pareto_gate": tune_pareto,
            }
        )
    if not ranked:
        return {
            "score": math.inf,
            "accepted_items": 0,
            "errors": 0,
            "coverage": 0.0,
            "risk": math.inf,
            "cost": math.inf,
            "allocation": [],
            "tune_total_items": total_items,
            "target_accepts": target_accepts,
            "tune_pareto_gate": False,
        }
    ranked.sort(key=lambda item: item["score"])
    return ranked[0]


def evaluate_tune_budgeted_cost_compression(
    rows: list[dict[str, str]],
    model_spec: dict[str, Any],
    *,
    baseline_coverage: float,
    baseline_risk: float,
    baseline_cost: float,
    cost_penalty: float,
    cheap_delta: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repeats = sorted({row["repeat_id"] for row in rows})
    tune_curves = {
        repeat_id: _item_curve(ranked_torch_items(rows, repeat_id, "tune", model_spec, cost_penalty, cheap_delta))
        for repeat_id in repeats
    }
    allocation = _tune_budget_allocation(
        tune_curves,
        baseline_coverage=baseline_coverage,
        baseline_risk=baseline_risk,
        baseline_cost=baseline_cost,
    )
    counts_by_repeat = {item["repeat_id"]: int(item["accepted_items"]) for item in allocation["allocation"]}

    decisions: list[dict[str, Any]] = []
    specs: list[dict[str, Any]] = [
        {
            "family": "gpu_torch_tune_budget_cost_compressed",
            "cost_penalty": cost_penalty,
            "cheap_delta": cheap_delta,
            "tune_allocation": allocation,
            "counts_by_repeat": counts_by_repeat,
        }
    ]
    for repeat_id in repeats:
        items = ranked_torch_items(rows, repeat_id, "holdout", model_spec, cost_penalty, cheap_delta)
        accepted_count = min(len(items), counts_by_repeat.get(repeat_id, 0))
        accepted_keys = {(item["domain"], item["item_id"]) for item in items[:accepted_count]}
        chosen = {(item["domain"], item["item_id"]): item for item in items[:accepted_count]}
        for item in items:
            key = (item["domain"], item["item_id"])
            accepted = key in accepted_keys
            decision: dict[str, Any] = {
                "accepted": accepted,
                "repeat_id": repeat_id,
                "domain": item["domain"],
                "item_id": item["item_id"],
                "policy": "gpu_torch_tune_budget_cost_compressed",
            }
            if accepted:
                selected = chosen[key]
                decision.update(
                    {
                        "action": selected["action"],
                        "score": selected["score"],
                        "correct": selected["correct"],
                        "cost": selected["cost"],
                    }
                )
            decisions.append(decision)
    return decisions, specs


def summarize_decisions(policy_name: str, decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for domain in ["overall", *sorted({row["domain"] for row in decisions})]:
        subset = decisions if domain == "overall" else [row for row in decisions if row["domain"] == domain]
        m = metrics(subset)
        rows.append({"policy": policy_name, "domain": domain, **m})
    return rows


def search(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_actions_from_prepare()
    baseline_rows = read_csv(dep_dir("prepare_sources") / "baseline_reference.csv")
    baseline_overall = next(
        (row for row in baseline_rows if row.get("method") == "matched_verifier_threshold_baseline" and row.get("domain") == "overall"),
        {},
    )
    baseline_by_domain = {
        row["domain"]: as_float(row.get("coverage_mean"), as_float(row.get("coverage_overall")))
        for row in baseline_rows
        if row.get("method") == "matched_verifier_threshold_baseline"
    }
    target_by_domain = {domain: max(0.05, cov) for domain, cov in baseline_by_domain.items()}
    target_by_domain.setdefault("overall", max(0.25, as_float(baseline_overall.get("coverage_mean"), 0.4)))
    baseline_risk = as_float(baseline_overall.get("risk_mean"), as_float(baseline_overall.get("risk_overall"), 1.0))
    baseline_cov = as_float(baseline_overall.get("coverage_mean"), as_float(baseline_overall.get("coverage_overall"), 0.0))
    baseline_cost = as_float(baseline_overall.get("cost_mean"), as_float(baseline_overall.get("cost_overall"), 0.0))

    candidates: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = []
    for family, domain_specific, weights in [
        ("verifier_threshold", False, None),
        ("verifier_threshold", True, None),
        ("confidence_threshold", True, None),
        ("agreement_threshold", True, None),
        ("cost_aware_meta_score", True, {"verifier": 0.55, "confidence": 0.2, "agreement": 0.2, "cost": 0.12}),
        ("cost_aware_meta_score", True, {"verifier": 0.35, "confidence": 0.25, "agreement": 0.25, "candidate_count": 0.08, "cost": 0.18}),
        ("cost_aware_meta_score", True, {"verifier": 0.7, "confidence": 0.1, "agreement": 0.1, "cost": 0.04}),
    ]:
        decisions, specs = evaluate_threshold_family(rows, family, target_by_domain, weights=weights, domain_specific=domain_specific)
        name = family + ("_domain" if domain_specific else "_global")
        if weights:
            name += "_" + hashlib.sha1(json.dumps(weights, sort_keys=True).encode()).hexdigest()[:6]
        for row in decisions:
            row["policy"] = name
        candidates.append((name, decisions, specs))

    train_torch_meta(rows, out)
    model_path = out / "torch_meta_model.json"
    if model_path.exists():
        model_spec = json.loads(model_path.read_text(encoding="utf-8"))
        decisions, specs = evaluate_torch_meta(rows, model_spec, target_by_domain)
        candidates.append(("gpu_torch_meta_risk", decisions, specs))
        for target in [
            max(0.05, target_by_domain["overall"] - 0.02),
            target_by_domain["overall"],
            min(0.95, target_by_domain["overall"] + 0.04),
        ]:
            for penalty in [0.0, 0.01, 0.03, 0.05, 0.1]:
                decisions, specs = evaluate_torch_budgeted_rank(rows, model_spec, target, penalty)
                name = f"gpu_torch_budgeted_rank_cov{target:.3f}_p{penalty:.2f}".replace(".", "p")
                for row in decisions:
                    row["policy"] = name
                candidates.append((name, decisions, specs))
        for penalty in [i / 100 for i in range(0, 31)] + [0.4, 0.5, 0.75, 1.0, 1.5, 2.0]:
            for cheap_delta in [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]:
                decisions, specs = evaluate_tune_budgeted_cost_compression(
                    rows,
                    model_spec,
                    baseline_coverage=target_by_domain["overall"],
                    baseline_risk=baseline_risk,
                    baseline_cost=baseline_cost,
                    cost_penalty=penalty,
                    cheap_delta=cheap_delta,
                )
                name = f"gpu_torch_tune_budget_compressed_p{penalty:.3f}_d{cheap_delta:.3f}".replace(".", "p")
                for row in decisions:
                    row["policy"] = name
                candidates.append((name, decisions, specs))

    summary_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    specs_by_policy: dict[str, Any] = {}
    for name, decisions, specs in candidates:
        summary_rows.extend(summarize_decisions(name, decisions))
        decision_rows.extend(decisions)
        specs_by_policy[name] = specs

    baseline_risk = as_float(baseline_overall.get("risk_mean"), as_float(baseline_overall.get("risk_overall"), 1.0))
    baseline_cov = as_float(baseline_overall.get("coverage_mean"), as_float(baseline_overall.get("coverage_overall"), 0.0))
    baseline_cost = as_float(baseline_overall.get("cost_mean"), as_float(baseline_overall.get("cost_overall"), 0.0))
    baseline_accepts = as_float(baseline_overall.get("accepted_total"), baseline_cov * 150.0)
    baseline_errors = as_float(baseline_overall.get("accepted_errors_total"), baseline_risk * baseline_accepts)
    baseline_cost_total = baseline_cost * 150.0
    scored = []
    pareto_rows = []
    for row in summary_rows:
        if row["domain"] != "overall":
            continue
        risk = as_float(row["risk"], 1.0)
        coverage = as_float(row["coverage"])
        cost = as_float(row["cost"])
        accepted_count = as_float(row.get("accepted_count"))
        accepted_errors = as_float(row.get("accepted_errors"))
        gate = {
            "coverage_gap_vs_baseline": coverage - baseline_cov,
            "risk_delta_vs_baseline": risk - baseline_risk,
            "cost_delta_vs_baseline": cost - baseline_cost,
            "cost_ratio_vs_baseline": cost / baseline_cost if baseline_cost else math.inf,
            "accepted_delta_vs_baseline": accepted_count - baseline_accepts,
            "accepted_error_delta_vs_baseline": accepted_errors - baseline_errors,
            "total_cost_delta_vs_baseline": cost * as_float(row.get("n_items"), 0.0) - baseline_cost_total,
            "beats_baseline": coverage >= baseline_cov - 0.02 - 1e-9 and risk <= baseline_risk * 0.75 + 1e-6,
            "pareto_dominates_baseline": coverage >= baseline_cov - 1e-9 and risk <= baseline_risk + 1e-9 and cost <= baseline_cost + 1e-9,
        }
        if gate["pareto_dominates_baseline"]:
            pareto_rows.append({**row, **gate})
        if gate["pareto_dominates_baseline"]:
            score = -2.0 + risk - max(0.0, coverage - baseline_cov) * 0.2 + cost * 0.02
        elif gate["beats_baseline"]:
            score = (
                -1.0
                + risk
                + max(0.0, cost - baseline_cost) * 0.4
                + max(0.0, baseline_cov - coverage) * 0.2
                - max(0.0, coverage - baseline_cov) * 0.2
            )
        else:
            score = risk + max(0.0, baseline_cov - coverage) * 1.5 + max(0.0, cost - baseline_cost) * 0.1
        scored.append((score, row["policy"], row, gate))
    scored.sort(key=lambda item: item[0])
    best = scored[0]

    write_csv(out / "policy_search_results.csv", summary_rows)
    write_csv(out / "policy_decisions_holdout.csv", decision_rows)
    write_json(
        out / "cost_compression_report.json",
        {
            "schema_version": "aira.ttc_cost_compression_report.v1",
            "objective": "Find a policy with coverage >= baseline, risk <= baseline, and cost <= baseline.",
            "baseline": {
                "method": "matched_verifier_threshold_baseline",
                "coverage": baseline_cov,
                "risk": baseline_risk,
                "cost": baseline_cost,
                "accepted_total": baseline_accepts,
                "accepted_errors_total": baseline_errors,
                "cost_total": baseline_cost_total,
            },
            "pareto_candidate_count": len(pareto_rows),
            "pareto_candidates": pareto_rows[:25],
        },
    )
    write_json(
        out / "best_policy_spec.json",
        {
            "schema_version": "aira.ttc_best_policy_spec.v1",
            "selected_policy": best[1],
            "selection_score": best[0],
            "overall_metrics": best[2],
            "baseline": {
                "method": "matched_verifier_threshold_baseline",
                "coverage": baseline_cov,
                "risk": baseline_risk,
                "cost": baseline_cost,
            },
            "gate": best[3],
            "threshold_specs": specs_by_policy.get(best[1], []),
            "all_policy_specs": specs_by_policy,
        },
    )


def synthesize(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    search_dir = dep_dir("search_policies")
    results = read_csv(search_dir / "policy_search_results.csv")
    best = json.loads((search_dir / "best_policy_spec.json").read_text(encoding="utf-8"))
    overall = [row for row in results if row["domain"] == "overall"]
    frontier_rows = []
    for row in overall:
        frontier_rows.append(
            {
                "policy": row["policy"],
                "coverage": row["coverage"],
                "risk": row["risk"],
                "cost": row["cost"],
                "accepted_count": row["accepted_count"],
                "accepted_errors": row["accepted_errors"],
            }
        )
    write_csv(out / "frontier_summary.csv", frontier_rows)
    gate = best["gate"]
    if gate["pareto_dominates_baseline"]:
        status = "candidate_found_pareto"
    elif gate["beats_baseline"]:
        status = "candidate_found_tradeoff"
    else:
        status = "candidate_not_found"
    claim_gate = {
        "schema_version": "aira.ttc_policy_claim_gate.v1",
        "status": status,
        "selected_policy": best["selected_policy"],
        "baseline": best["baseline"],
        "selected_policy_metrics": best["overall_metrics"],
        "gate": gate,
        "interpretation": (
            "A policy candidate Pareto-dominated the matched threshold baseline on this screening ledger."
            if status == "candidate_found_pareto"
            else "A policy candidate improved the risk-coverage gate but did not Pareto-dominate the matched threshold baseline."
            if status == "candidate_found_tradeoff"
            else "No searched policy beat the strong matched verifier threshold baseline; new evidence supports changing the research direction or adding richer signals/fresh data."
        ),
        "next_required_work": [
            "If candidate_found_pareto, run fresh locked-holdout evaluation with larger unique-item budget.",
            "If candidate_found_tradeoff, prioritize compute-cost compression and matched-cost comparison before making a positive paper claim.",
            "If candidate_not_found, add richer uncertainty signals or stronger verifiers before drafting a positive method paper.",
            "Preserve matched coverage and matched cost comparisons as hard claim gates.",
        ],
    }
    write_json(out / "claim_gate.json", claim_gate)
    if status == "candidate_found_pareto":
        recommended_next_loop = "fresh_locked_holdout"
    elif status == "candidate_found_tradeoff":
        recommended_next_loop = "cost_compression_then_fresh_eval"
    else:
        recommended_next_loop = "feature_acquisition"
    write_json(
        out / "next_round_plan.json",
        {
            "schema_version": "aira.ttc_next_round_plan.v1",
            "time_budget_hours": 4,
            "gpu_budget_fraction": 0.8,
            "recommended_next_loop": recommended_next_loop,
            "tasks": [
                "Lock canonical evaluation ledger and baseline operating points.",
                "Add self-consistency entropy, margin, answer validity, and domain-specific process-verifier signals.",
                "For tradeoff candidates, reduce compute cost before claiming a win against the strong threshold baseline.",
                "Run fresh evaluation only for candidates that pass replay screening.",
            ],
        },
    )
    report = [
        "# AIRA TTC Policy Search Loop",
        "",
        f"Status: {status}",
        f"Selected policy: `{best['selected_policy']}`",
        "",
        "## Baseline",
        json.dumps(best["baseline"], indent=2, sort_keys=True),
        "",
        "## Selected Policy Metrics",
        json.dumps(best["overall_metrics"], indent=2, sort_keys=True),
        "",
        "## Gate",
        json.dumps(gate, indent=2, sort_keys=True),
        "",
        "## Interpretation",
        claim_gate["interpretation"],
        "",
    ]
    (out / "experiment_loop_report.md").write_text("\n".join(report), encoding="utf-8")
    (out / "aira_policy_search_writing_brief.md").write_text(
        "\n".join(
            [
                "# Writing Brief",
                "",
                "This AIRA bundle searches for a new policy that can beat a strong matched threshold baseline.",
                f"Claim gate status: `{status}`.",
                "Do not draft a positive method paper unless the claim gate is `candidate_found_pareto`, or a tradeoff candidate is confirmed under matched-cost fresh locked-holdout evaluation.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "search", "synthesize"])
    parser.add_argument("--out", default=".")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "search":
        search(args)
    elif args.command == "synthesize":
        synthesize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
