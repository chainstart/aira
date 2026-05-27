"""Production experiment runner for AIRA."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from aira import __version__
from aira.bundles import BUNDLE_SCHEMA_VERSION, validate_bundle, write_json


RUNNER_SCHEMA_VERSION = "aira.production_runner.v1"
PLAN_SCHEMA_VERSION = "aira.production_plan.v1"
PROFILE_SCHEMA_VERSION = "aira.production_profile.v1"
CREATED_AT = "2026-05-20T00:00:00Z"
TASK_ID = "AIRA-PROD-RUNNER-001"
RUNNER_MODEL_ID = "production-local-controlled-python-runner-v1"
RUNNER_DATASET_ID = "operator-supplied-production-plan"
COMMAND = "python3 -m aira experiments run"

DESTRUCTIVE_DENY_PATTERNS = (
    "rm -rf",
    "git reset --hard",
    "git checkout --",
    "git clean -fd",
    "PRIVATE_KEY",
    "MNEMONIC",
    "withdraw",
)
STRICT_DENY_PATTERNS = (
    *DESTRUCTIVE_DENY_PATTERNS,
    "curl ",
    "wget ",
    "pip install",
)
STRICT_DENIED_IMPORT_ROOTS = {
    "ftplib",
    "http",
    "multiprocessing",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "threading",
    "urllib",
}
STRICT_DENIED_CALLS = {
    "eval",
    "exec",
    "compile",
    "os.popen",
    "os.spawnl",
    "os.spawnle",
    "os.spawnlp",
    "os.spawnlpe",
    "os.spawnv",
    "os.spawnve",
    "os.spawnvp",
    "os.spawnvpe",
    "os.system",
}
TASK_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class ProductionProfile:
    schema_version: str
    name: str
    max_tasks: int
    task_timeout_seconds: int
    max_stdout_bytes: int
    max_stderr_bytes: int
    max_output_files_per_task: int
    max_cpu_threads: int
    allowed_command_kinds: tuple[str, ...]
    allowed_packages: tuple[str, ...]
    network_policy: str
    live_model_calls: bool
    gpu_required: bool
    external_datasets_required: bool
    package_installation: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "max_tasks": self.max_tasks,
            "task_timeout_seconds": self.task_timeout_seconds,
            "max_stdout_bytes": self.max_stdout_bytes,
            "max_stderr_bytes": self.max_stderr_bytes,
            "max_output_files_per_task": self.max_output_files_per_task,
            "max_cpu_threads": self.max_cpu_threads,
            "allowed_command_kinds": list(self.allowed_command_kinds),
            "allowed_packages": list(self.allowed_packages),
            "network_policy": self.network_policy,
            "live_model_calls": self.live_model_calls,
            "gpu_required": self.gpu_required,
            "external_datasets_required": self.external_datasets_required,
            "package_installation": self.package_installation,
        }


@dataclass
class PolicyReport:
    allowed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)

    def fail(self, check_id: str, message: str) -> None:
        self.allowed = False
        self.errors.append(message)
        self.checks.append({"id": check_id, "status": "fail", "message": message})

    def pass_check(self, check_id: str, message: str) -> None:
        self.checks.append({"id": check_id, "status": "pass", "message": message})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "aira.production_policy_report.v1",
            "allowed": self.allowed,
            "errors": self.errors,
            "warnings": self.warnings,
            "checks": self.checks,
        }


def production_local_profile() -> ProductionProfile:
    return ProductionProfile(
        schema_version=PROFILE_SCHEMA_VERSION,
        name="production-local",
        max_tasks=8,
        task_timeout_seconds=30,
        max_stdout_bytes=20_000,
        max_stderr_bytes=20_000,
        max_output_files_per_task=12,
        max_cpu_threads=2,
        allowed_command_kinds=("inline_python",),
        allowed_packages=(),
        network_policy="none",
        live_model_calls=False,
        gpu_required=False,
        external_datasets_required=False,
        package_installation=False,
    )


def production_open_profile() -> ProductionProfile:
    return ProductionProfile(
        schema_version=PROFILE_SCHEMA_VERSION,
        name="production-open",
        max_tasks=64,
        task_timeout_seconds=24 * 3600,
        max_stdout_bytes=500_000,
        max_stderr_bytes=500_000,
        max_output_files_per_task=256,
        max_cpu_threads=max(1, os.cpu_count() or 1),
        allowed_command_kinds=("inline_python", "external_command", "shell_command", "command"),
        allowed_packages=("*",),
        network_policy="unrestricted",
        live_model_calls=True,
        gpu_required=True,
        external_datasets_required=True,
        package_installation=True,
    )


def load_profile(name: str) -> ProductionProfile:
    if name == "production-local":
        return production_local_profile()
    if name == "production-open":
        return production_open_profile()
    raise ValueError("Unsupported AIRA experiment profile. Use --profile production-local or --profile production-open.")


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value.strip()) and not path.is_absolute() and ".." not in path.parts


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 80] + "\n[AIRA output truncated]\n"


def _requested_packages(plan: dict[str, Any]) -> list[str]:
    resource_requirements = plan.get("resource_requirements", {})
    packages = []
    if isinstance(resource_requirements, dict):
        packages.extend(resource_requirements.get("python_packages", []) or [])
    packages.extend(plan.get("python_packages", []) or [])
    return [str(package).strip() for package in packages if str(package).strip()]


def _import_roots(code: str) -> set[str]:
    roots: set[str] = set()
    try:
        parsed = ast.parse(code)
    except SyntaxError:
        return roots
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _denied_calls(code: str, denied_calls: set[str]) -> list[str]:
    try:
        parsed = ast.parse(code)
    except SyntaxError:
        return []
    calls: set[str] = set()
    for node in ast.walk(parsed):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in denied_calls:
                calls.add(name)
    return sorted(calls)


def _profile_allows(flag: str, profile: ProductionProfile) -> bool:
    return {
        "network_required": profile.network_policy == "unrestricted",
        "external_datasets_required": profile.external_datasets_required,
        "gpu_required": profile.gpu_required,
        "live_model_calls": profile.live_model_calls,
    }.get(flag, False)


def _normalize_package_name(package: str) -> str:
    return (
        package.split("[", 1)[0]
        .split("==", 1)[0]
        .split("!=", 1)[0]
        .split(">=", 1)[0]
        .split("<=", 1)[0]
        .split("~=", 1)[0]
        .split(">", 1)[0]
        .split("<", 1)[0]
        .lower()
    )


def _runner_model_id(profile: ProductionProfile) -> str:
    if profile.name == "production-open":
        return "production-open-python-runner-v1"
    return RUNNER_MODEL_ID


def _command_text(command: dict[str, Any]) -> str:
    if isinstance(command.get("code"), str):
        return str(command["code"])
    if isinstance(command.get("argv"), list):
        return " ".join(str(part) for part in command["argv"])
    return str(command.get("command") or "")


def _validate_task_graph(tasks: list[dict[str, Any]], policy: PolicyReport) -> list[str]:
    task_map: dict[str, dict[str, Any]] = {}
    ordered_ids: list[str] = []
    for index, task in enumerate(tasks):
        task_id = str(task.get("id", "")).strip()
        if not TASK_ID_RE.match(task_id):
            policy.fail("task_id", f"tasks[{index}].id must contain only letters, numbers, '.', '_', or '-'.")
            continue
        if task_id in task_map:
            policy.fail("task_id_unique", f"Duplicate task id: {task_id}.")
            continue
        task_map[task_id] = task
        ordered_ids.append(task_id)

    for task in tasks:
        task_id = str(task.get("id", "")).strip()
        dependencies = task.get("dependencies", []) or []
        if not isinstance(dependencies, list):
            policy.fail("dependencies", f"Task {task_id} dependencies must be a list.")
            continue
        for dep in dependencies:
            dep_id = str(dep).strip()
            if dep_id not in task_map:
                policy.fail("dependencies", f"Task {task_id} references missing dependency {dep_id}.")

    visiting: set[str] = set()
    visited: set[str] = set()
    topo: list[str] = []

    def visit(task_id: str, trail: list[str]) -> None:
        if task_id in visited:
            return
        if task_id in visiting:
            policy.fail("dependencies", "Dependency cycle: " + " -> ".join(trail + [task_id]))
            return
        visiting.add(task_id)
        for dep in task_map[task_id].get("dependencies", []) or []:
            dep_id = str(dep).strip()
            if dep_id in task_map:
                visit(dep_id, trail + [task_id])
        visiting.remove(task_id)
        visited.add(task_id)
        topo.append(task_id)

    for task_id in ordered_ids:
        if task_id in task_map:
            visit(task_id, [])
    return topo


def evaluate_production_policy(plan: dict[str, Any], profile: ProductionProfile) -> tuple[PolicyReport, list[str]]:
    policy = PolicyReport()
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        policy.fail("plan_schema", f"Plan schema_version must be {PLAN_SCHEMA_VERSION}.")
    else:
        policy.pass_check("plan_schema", "Plan schema is supported.")

    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        policy.fail("tasks", "Plan must contain a non-empty tasks list.")
        return policy, []
    if len(tasks) > profile.max_tasks:
        policy.fail("tasks", f"Plan has {len(tasks)} tasks, above profile limit {profile.max_tasks}.")
    else:
        policy.pass_check("tasks", "Plan task count is within the profile limit.")

    gated_flags = ("network_required", "external_datasets_required", "gpu_required", "live_model_calls")
    blocked_flags: list[str] = []
    for flag in gated_flags:
        if plan.get(flag) is True and not _profile_allows(flag, profile):
            blocked_flags.append(flag)
            policy.fail(flag, f"Plan field {flag} is not allowed by profile {profile.name}.")
    if not blocked_flags:
        if profile.name == "production-open":
            policy.pass_check(
                "profile_gates",
                "Production-open profile permits network access, external datasets, GPU use, and live model calls when requested.",
            )
        else:
            policy.pass_check("profile_gates", "Production-local profile disables network, external datasets, GPU, and live calls.")

    allowed_packages = {package.lower() for package in profile.allowed_packages}
    wildcard_packages = "*" in allowed_packages
    for package in _requested_packages(plan):
        name = _normalize_package_name(package)
        if not wildcard_packages and name not in allowed_packages:
            policy.fail("packages", f"Package is not allowed by profile {profile.name}: {package}.")
    if _requested_packages(plan) and wildcard_packages and profile.package_installation:
        policy.pass_check("packages", "Production-open profile allows requested Python package installation.")
    elif not _requested_packages(plan):
        policy.pass_check("packages", "Plan does not request package installation.")

    topo = _validate_task_graph(tasks, policy)
    seen_artifact_ids: set[str] = set()
    for task in tasks:
        task_id = str(task.get("id", "")).strip()
        command = task.get("command")
        if not isinstance(command, dict):
            policy.fail("command", f"Task {task_id} command must be an object.")
            continue
        kind = str(command.get("kind", "")).strip()
        if kind not in profile.allowed_command_kinds:
            policy.fail("command_kind", f"Task {task_id} command kind is not allowed: {kind}.")
            continue
        command_text = _command_text(command)
        deny_patterns = DESTRUCTIVE_DENY_PATTERNS if profile.name == "production-open" else STRICT_DENY_PATTERNS
        for pattern in deny_patterns:
            if pattern in command_text:
                policy.fail("deny_patterns", f"Task {task_id} command contains denied pattern: {pattern}.")
        if kind == "inline_python":
            code = str(command.get("code", ""))
            try:
                ast.parse(code)
            except SyntaxError as exc:
                policy.fail("python_syntax", f"Task {task_id} inline Python is not syntactically valid: {exc}.")
            denied_imports = sorted(_import_roots(code) & (set() if profile.name == "production-open" else STRICT_DENIED_IMPORT_ROOTS))
            if denied_imports:
                policy.fail("imports", f"Task {task_id} imports denied modules: {denied_imports}.")
            denied_calls = _denied_calls(code, set() if profile.name == "production-open" else STRICT_DENIED_CALLS)
            if denied_calls:
                policy.fail("script_calls", f"Task {task_id} calls denied functions: {denied_calls}.")
        elif not (command.get("command") or command.get("argv")):
            policy.fail("command", f"Task {task_id} external command must provide command or argv.")
        outputs = task.get("outputs", [])
        if not isinstance(outputs, list) or not outputs:
            policy.fail("outputs", f"Task {task_id} must declare at least one output artifact.")
            continue
        if len(outputs) > profile.max_output_files_per_task:
            policy.fail("outputs", f"Task {task_id} declares too many outputs.")
        for output in outputs:
            if not isinstance(output, dict):
                policy.fail("outputs", f"Task {task_id} output entries must be objects.")
                continue
            artifact_id = str(output.get("artifact_id", "")).strip()
            output_path = str(output.get("path", "")).strip()
            if not artifact_id:
                policy.fail("outputs", f"Task {task_id} output artifact_id is required.")
            elif artifact_id in seen_artifact_ids:
                policy.fail("outputs", f"Duplicate output artifact_id: {artifact_id}.")
            seen_artifact_ids.add(artifact_id)
            if not _safe_relative_path(output_path):
                policy.fail("outputs", f"Task {task_id} output path is not safe: {output_path}.")
    if policy.allowed:
        policy.pass_check("script_controls", "All tasks use command kinds allowed by the selected profile with declared outputs.")
    return policy, topo


def _bounded_env(profile: ProductionProfile, out: Path, task_dir: Path, dep_dirs: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env["AIRA_PROFILE"] = profile.name
    env["AIRA_NETWORK_POLICY"] = profile.network_policy
    env["AIRA_TASK_DIR"] = str(task_dir)
    env["AIRA_OUTPUT_DIR"] = str(out)
    env["AIRA_DEP_DIRS"] = json.dumps(dep_dirs, sort_keys=True)
    env["AIRA_CPU_THREADS"] = str(profile.max_cpu_threads)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "PYTORCH_NUM_THREADS",
    ):
        env[key] = str(profile.max_cpu_threads)
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def _execute_task(
    task: dict[str, Any],
    *,
    out: Path,
    profile: ProductionProfile,
    task_dirs: dict[str, str],
) -> dict[str, Any]:
    task_id = str(task["id"])
    task_dir = out / "work" / "tasks" / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    dep_dirs = {dep: task_dirs[dep] for dep in task.get("dependencies", []) or [] if dep in task_dirs}
    command = task["command"]
    kind = str(command.get("kind", "inline_python"))
    if kind == "inline_python":
        script_path = task_dir / "run.py"
        script_path.write_text(str(command["code"]).rstrip() + "\n", encoding="utf-8")
        argv = [sys.executable, str(script_path)]
        command_display = " ".join(argv)
    else:
        script_path = None
        if isinstance(command.get("argv"), list):
            argv = [str(part) for part in command["argv"]]
        else:
            argv = shlex.split(str(command.get("command") or ""))
        command_display = " ".join(argv)
    result = subprocess.run(
        argv,
        cwd=task_dir,
        env=_bounded_env(profile, out, task_dir, dep_dirs),
        capture_output=True,
        text=True,
        timeout=profile.task_timeout_seconds,
    )
    return {
        "task_id": task_id,
        "status": "passed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "timed_out": False,
        "stdout": _truncate(result.stdout, profile.max_stdout_bytes),
        "stderr": _truncate(result.stderr, profile.max_stderr_bytes),
        "script_path": str(script_path) if script_path else None,
        "command": command_display,
        "command_kind": kind,
        "task_dir": str(task_dir),
    }


def _materialize_outputs(task: dict[str, Any], task_record: dict[str, Any], out: Path) -> list[dict[str, Any]]:
    task_id = str(task["id"])
    task_dir = Path(task_record["task_dir"])
    materialized: list[dict[str, Any]] = []
    missing: list[str] = []
    for output in task.get("outputs", []):
        relative = str(output["path"])
        source = task_dir / relative
        if not source.is_file():
            missing.append(relative)
            continue
        destination = out / "artifacts" / "tasks" / task_id / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        materialized.append(
            {
                "artifact_id": str(output["artifact_id"]),
                "path": destination.relative_to(out).as_posix(),
                "kind": str(output.get("kind", "task_output")),
                "description": str(output.get("description", f"Output from task {task_id}.")),
                "source_task_id": task_id,
                "source_path": relative,
                "sha256": _file_sha256(destination),
                "size_bytes": destination.stat().st_size,
            }
        )
    if missing:
        task_record["status"] = "failed"
        task_record["missing_outputs"] = missing
        task_record["stderr"] = (task_record.get("stderr") or "") + "\nMissing declared outputs: " + ", ".join(missing)
    task_record["materialized_outputs"] = materialized
    return materialized


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_plan(plan_path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(plan_path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("AIRA production plan must be a JSON object.")
    return data


def _install_requested_packages(plan: dict[str, Any], profile: ProductionProfile, policy: PolicyReport, out: Path) -> None:
    packages = _requested_packages(plan)
    if not packages or not profile.package_installation:
        return
    install_dir = out / "work" / "package_install"
    install_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", *packages]
    try:
        proc = subprocess.run(
            cmd,
            cwd=install_dir,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired as exc:
        (install_dir / "stdout.txt").write_text(exc.stdout or "", encoding="utf-8", errors="ignore")
        (install_dir / "stderr.txt").write_text(exc.stderr or "pip install timed out", encoding="utf-8", errors="ignore")
        policy.fail("package_installation", "Requested package installation timed out.")
        return
    (install_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8", errors="ignore")
    (install_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8", errors="ignore")
    if proc.returncode == 0:
        policy.pass_check("package_installation", f"Installed requested Python packages: {packages}.")
    else:
        policy.fail("package_installation", f"Package installation failed with return code {proc.returncode}.")


def _execution_flags(plan: dict[str, Any], profile: ProductionProfile) -> dict[str, bool]:
    return {
        "network_required": bool(plan.get("network_required")) or profile.network_policy == "unrestricted",
        "external_datasets_required": bool(plan.get("external_datasets_required")) or profile.external_datasets_required,
        "gpu_required": bool(plan.get("gpu_required")) or profile.gpu_required,
        "live_model_calls": bool(plan.get("live_model_calls")) or profile.live_model_calls,
        "package_installation": bool(_requested_packages(plan)) and profile.package_installation,
    }


def _profile_limitations(profile: ProductionProfile, flags: dict[str, bool]) -> list[str]:
    if profile.name == "production-open":
        return [
            "The production-open profile intentionally permits package installation, network downloads, external datasets, GPU execution, and live model/API calls when requested by the plan.",
            "Reproducibility depends on recorded package versions, external data/model fingerprints, API/model versions, and operator-provided license or credential attestations.",
            "AIRA still records policy, execution trace, provenance, materialized artifacts, and run ledger entries, but open-profile runs are not deterministic by default.",
        ]
    return [
        "The production-local profile executes only explicitly declared local Python tasks.",
        "No package installation, network access, GPU execution, external datasets, or live model APIs are enabled.",
        "The runner provides local subprocess isolation and timeout bounds, not a container sandbox.",
    ]


def _plan_limitations(plan: dict[str, Any], profile: ProductionProfile, flags: dict[str, bool]) -> list[str]:
    limitations = [str(item).strip() for item in plan.get("limitations", []) or [] if str(item).strip()]
    limitations.extend(_profile_limitations(profile, flags))
    return list(dict.fromkeys(limitations))


def _plan_claims(plan: dict[str, Any], *, status: str, profile: ProductionProfile, limitations: list[str]) -> list[dict[str, Any]]:
    claims = plan.get("claims")
    if isinstance(claims, list) and claims:
        normalized: list[dict[str, Any]] = []
        for index, claim in enumerate(claims):
            if not isinstance(claim, dict):
                continue
            claim_id = str(claim.get("claim_id") or claim.get("id") or f"aira-plan-claim-{index + 1}").strip()
            claim_text = str(claim.get("claim") or claim.get("text") or "").strip()
            if not claim_id or not claim_text:
                continue
            item = dict(claim)
            item["claim_id"] = claim_id
            item["claim"] = claim_text
            item.setdefault("status", "confirmed" if status == "passed" else "observed")
            item.setdefault("reproduction_status", "reproduced" if status == "passed" else "failed")
            supported_by = item.get("supported_by", item.get("artifacts"))
            item["supported_by"] = [str(ref).strip() for ref in supported_by or [] if str(ref).strip()]
            item.setdefault("limitations", limitations)
            normalized.append(item)
        if normalized:
            return normalized

    claim_status = "confirmed" if status == "passed" else "observed"
    return [
        {
            "claim_id": "aira-production-runner-c1",
            "claim": (
                f"The AIRA {profile.name} runner executed a policy-checked experiment plan "
                "with explicit profile gating, resource accounting, failure isolation, and materialized artifacts."
            ),
            "status": claim_status,
            "reproduction_status": "reproduced" if status == "passed" else "failed",
            "supported_by": [
                "reproduction_status",
                "policy_report",
                "execution_trace",
                "task_summary",
                "provenance",
                "run_ledger_entry",
            ],
            "limitations": limitations,
        }
    ]


def _writing_brief(plan: dict[str, Any], profile: ProductionProfile) -> str:
    brief = str(plan.get("writing_brief_markdown") or plan.get("writing_brief") or "").strip()
    if brief:
        return brief.rstrip() + "\n"
    return "\n".join(
        [
            f"# AIRA {profile.name} Runner",
            "",
            f"This bundle records an AIRA `{profile.name}` experiment runner invocation.",
            "It is intended to move script execution responsibility from legacy ARA into AIRA while preserving artifact, provenance, and claim-boundary records.",
            "",
        ]
    )


def _write_bundle(
    *,
    out: Path,
    profile: ProductionProfile,
    plan: dict[str, Any],
    plan_path: Path,
    policy: PolicyReport,
    task_records: list[dict[str, Any]],
    materialized_outputs: list[dict[str, Any]],
    run_id: str,
) -> dict[str, Any]:
    artifacts_dir = out / "artifacts"
    memory_dir = out / "memory"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)

    plan_id = str(plan.get("plan_id", f"{profile.name}-plan"))
    flags = _execution_flags(plan, profile)
    model_id = _runner_model_id(profile)
    deterministic = not (
        flags["network_required"]
        or flags["external_datasets_required"]
        or flags["gpu_required"]
        or flags["live_model_calls"]
    )
    limitations = _plan_limitations(plan, profile, flags)
    status = "passed" if policy.allowed and all(task["status"] == "passed" for task in task_records) else "failed"
    task_summary = {
        "schema_version": "aira.production_task_summary.v1",
        "run_id": run_id,
        "plan_id": plan_id,
        "status": status,
        "task_count": len(task_records),
        "passed_task_count": sum(1 for task in task_records if task["status"] == "passed"),
        "failed_task_count": sum(1 for task in task_records if task["status"] == "failed"),
        "skipped_task_count": sum(1 for task in task_records if task["status"] == "skipped"),
        "tasks": task_records,
    }
    trace = {
        "schema_version": "aira.production_execution_trace.v1",
        "run_id": run_id,
        "profile": profile.to_dict(),
        "plan_path": str(plan_path),
        "task_order": [task["task_id"] for task in task_records],
        "tasks": task_records,
    }
    provenance = {
        "schema_version": "aira.benchmark_provenance.v1",
        "run_id": run_id,
        "task_id": TASK_ID,
        "created_at": CREATED_AT,
        "benchmark_id": plan_id,
        "dataset_id": RUNNER_DATASET_ID,
        "model_id": model_id,
        "input_fingerprints": {
            "dataset_sha256": _canonical_digest(plan.get("tasks", [])),
            "model_config_sha256": _canonical_digest(profile.to_dict()),
            "registry_snapshot_sha256": _canonical_digest({"plan_path": str(plan_path), "plan": plan}),
        },
        "execution": {
            "runner": "aira.production_runner.run_production_experiment",
            "command": f"{COMMAND} --profile {profile.name} --plan {plan_path} --out <bundle>",
            "package_version": __version__,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "determinism": {
            "deterministic": deterministic,
            "random_seed": None,
            "network_required": flags["network_required"],
            "external_datasets_required": flags["external_datasets_required"],
            "gpu_required": flags["gpu_required"],
            "live_model_calls": flags["live_model_calls"],
            "package_installation": flags["package_installation"],
        },
        "limitations": limitations,
    }
    ledger_entry = {
        "schema_version": "aira.run_ledger_entry.v1",
        "run_id": run_id,
        "task_id": TASK_ID,
        "created_at": CREATED_AT,
        "status": status,
        "bundle_path": str(out),
        "bundle_type": "aira_result_bundle",
        "benchmark_id": plan_id,
        "dataset_id": RUNNER_DATASET_ID,
        "model_id": model_id,
        "metrics": {
            "task_count": task_summary["task_count"],
            "passed_task_count": task_summary["passed_task_count"],
            "failed_task_count": task_summary["failed_task_count"],
            "skipped_task_count": task_summary["skipped_task_count"],
        },
        "provenance": {
            "path": "artifacts/provenance.json",
            "dataset_sha256": provenance["input_fingerprints"]["dataset_sha256"],
            "model_config_sha256": provenance["input_fingerprints"]["model_config_sha256"],
        },
        "reproducibility": {
            "deterministic": deterministic,
            "network_required": flags["network_required"],
            "external_datasets_required": flags["external_datasets_required"],
            "gpu_required": flags["gpu_required"],
            "live_model_calls": flags["live_model_calls"],
            "package_installation": flags["package_installation"],
            "command": f"{COMMAND} --profile {profile.name} --plan {plan_path} --out <bundle>",
        },
        "artifacts": [
            "artifacts/production_plan.json",
            "artifacts/policy_report.json",
            "artifacts/execution_trace.json",
            "artifacts/task_summary.json",
            "artifacts/provenance.json",
            "artifacts/reproduction_status.json",
            "memory/run_ledger.jsonl",
        ],
    }
    reproduction_status = {
        "schema_version": "aira.reproduction_status.v1",
        "status": "reproduced" if status == "passed" else "failed",
        "run_id": run_id,
        "task_id": TASK_ID,
        "benchmark_id": plan_id,
        "deterministic": deterministic,
        "network_required": flags["network_required"],
        "external_datasets_required": flags["external_datasets_required"],
        "gpu_required": flags["gpu_required"],
        "live_model_calls": flags["live_model_calls"],
        "package_installation": flags["package_installation"],
        "command": f"{COMMAND} --profile {profile.name} --plan {plan_path} --out <bundle>",
        "metrics": ledger_entry["metrics"],
    }

    write_json(artifacts_dir / "production_plan.json", plan)
    write_json(artifacts_dir / "policy_report.json", policy.to_dict())
    write_json(artifacts_dir / "execution_trace.json", trace)
    write_json(artifacts_dir / "task_summary.json", task_summary)
    write_json(artifacts_dir / "provenance.json", provenance)
    write_json(artifacts_dir / "reproduction_status.json", reproduction_status)
    write_json(artifacts_dir / "run_ledger_entry.json", ledger_entry)
    (memory_dir / "run_ledger.jsonl").write_text(json.dumps(ledger_entry, sort_keys=True) + "\n", encoding="utf-8")

    artifacts = [
        {
            "artifact_id": "production_plan",
            "path": "artifacts/production_plan.json",
            "kind": "production_plan",
            "description": f"Policy-checked {profile.name} experiment plan.",
        },
        {
            "artifact_id": "policy_report",
            "path": "artifacts/policy_report.json",
            "kind": "policy_report",
            "description": f"{profile.name} profile, package, command, and resource policy checks.",
        },
        {
            "artifact_id": "execution_trace",
            "path": "artifacts/execution_trace.json",
            "kind": "execution_trace",
            "description": "Bounded subprocess execution trace and failure isolation state.",
        },
        {
            "artifact_id": "task_summary",
            "path": "artifacts/task_summary.json",
            "kind": "task_summary",
            "description": "Machine-readable task pass/fail/skip summary.",
        },
        {
            "artifact_id": "provenance",
            "path": "artifacts/provenance.json",
            "kind": "provenance",
            "description": "Input, profile, runner, and reproducibility provenance.",
        },
        {
            "artifact_id": "reproduction_status",
            "path": "artifacts/reproduction_status.json",
            "kind": "reproduction_status",
            "description": f"Local reproduction status for the {profile.name} run.",
        },
        {
            "artifact_id": "run_ledger_entry",
            "path": "artifacts/run_ledger_entry.json",
            "kind": "run_ledger_entry",
            "description": f"Machine-readable {profile.name} runner ledger row.",
        },
        {
            "artifact_id": "run_ledger",
            "path": "memory/run_ledger.jsonl",
            "kind": "run_ledger",
            "description": "Bundle-local production runner ledger.",
        },
        *materialized_outputs,
    ]
    write_json(out / "artifact_manifest.json", {"artifacts": artifacts})
    write_json(
        out / "bundle_manifest.json",
        {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "bundle_type": "aira_result_bundle",
            "domain": "ai_ml",
            "created_at": CREATED_AT,
            "producer": "aira",
            "task_id": TASK_ID,
            "run_id": run_id,
            "benchmark_id": plan_id,
            "dataset_id": RUNNER_DATASET_ID,
            "model_id": model_id,
            "deterministic": deterministic,
            "network_required": flags["network_required"],
            "external_datasets_required": flags["external_datasets_required"],
            "gpu_required": flags["gpu_required"],
            "live_model_calls": flags["live_model_calls"],
            "package_installation": flags["package_installation"],
            "production_runner": {
                "schema_version": RUNNER_SCHEMA_VERSION,
                "profile": profile.name,
                "policy_artifact": "artifacts/policy_report.json",
                "trace_artifact": "artifacts/execution_trace.json",
            },
        },
    )
    write_json(
        out / "claims.json",
        {"claims": _plan_claims(plan, status=status, profile=profile, limitations=limitations)},
    )
    (out / "writing_brief.md").write_text(_writing_brief(plan, profile), encoding="utf-8")
    (out / "limitations.md").write_text(
        "\n".join(["# Limitations", "", *[f"- {item}" for item in limitations], ""]),
        encoding="utf-8",
    )
    validation = validate_bundle(out)
    return {
        "status": status,
        "task_summary": task_summary,
        "trace": trace,
        "provenance": provenance,
        "ledger_entry": ledger_entry,
        "validation": validation.to_dict(),
    }


def run_production_experiment(profile_name: str, plan_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    profile = load_profile(profile_name)
    plan_file = Path(plan_path).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan = _load_plan(plan_file)
    run_id = f"aira-prod-{_canonical_digest({'profile': profile.name, 'plan': plan})[:12]}"
    policy, task_order = evaluate_production_policy(plan, profile)

    task_records: list[dict[str, Any]] = []
    materialized_outputs: list[dict[str, Any]] = []
    task_dirs: dict[str, str] = {}
    status_by_task: dict[str, str] = {}
    task_map = {str(task["id"]): task for task in plan.get("tasks", []) if isinstance(task, dict) and "id" in task}

    if policy.allowed:
        _install_requested_packages(plan, profile, policy, out)
    if policy.allowed:
        for task_id in task_order:
            task = task_map[task_id]
            failed_deps = [
                dep
                for dep in task.get("dependencies", []) or []
                if status_by_task.get(str(dep)) != "passed"
            ]
            if failed_deps:
                record = {
                    "task_id": task_id,
                    "status": "skipped",
                    "skipped_reason": f"Dependency did not pass: {', '.join(failed_deps)}",
                    "returncode": None,
                    "timed_out": False,
                    "stdout": "",
                    "stderr": "",
                    "materialized_outputs": [],
                }
                task_records.append(record)
                status_by_task[task_id] = "skipped"
                continue
            try:
                record = _execute_task(task, out=out, profile=profile, task_dirs=task_dirs)
            except subprocess.TimeoutExpired as exc:
                task_dir = out / "work" / "tasks" / task_id
                record = {
                    "task_id": task_id,
                    "status": "failed",
                    "returncode": -1,
                    "timed_out": True,
                    "stdout": _truncate(exc.stdout or "", profile.max_stdout_bytes),
                    "stderr": "Timed out",
                    "script_path": str(task_dir / "run.py"),
                    "task_dir": str(task_dir),
                    "materialized_outputs": [],
                }
            if record["status"] == "passed":
                materialized_outputs.extend(_materialize_outputs(task, record, out))
            task_records.append(record)
            status_by_task[task_id] = str(record["status"])
            if "task_dir" in record:
                task_dirs[task_id] = str(record["task_dir"])
    else:
        for task_id in task_order:
            task_records.append(
                {
                    "task_id": task_id,
                    "status": "skipped",
                    "skipped_reason": "Policy checks failed before execution.",
                    "returncode": None,
                    "timed_out": False,
                    "stdout": "",
                    "stderr": "",
                    "materialized_outputs": [],
                }
            )

    bundle = _write_bundle(
        out=out,
        profile=profile,
        plan=plan,
        plan_path=plan_file,
        policy=policy,
        task_records=task_records,
        materialized_outputs=materialized_outputs,
        run_id=run_id,
    )
    status = "passed" if bundle["status"] == "passed" and bundle["validation"]["valid"] else "failed"
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "status": status,
        "profile": profile.name,
        "bundle_path": str(out),
        "run_id": run_id,
        "plan_path": str(plan_file),
        "policy": policy.to_dict(),
        "tasks": task_records,
        "materialized_artifacts": materialized_outputs,
        "validation": bundle["validation"],
    }
