import json

from aira import cli
from aira.migration import build_inventory


def _write_fake_ara_source(root):
    for relative in [
        "ara/agents",
        "ara/tools",
        "ara/labs",
        "scripts",
        "projects/fixture-project/exp",
        "projects/fixture-project/writing",
    ]:
        (root / relative).mkdir(parents=True, exist_ok=True)
    (root / "ara/agents/experiment.py").write_text("class ExperimentAgent:\n    pass\n", encoding="utf-8")
    (root / "ara/tools/code_executor.py").write_text("class CodeExecutor:\n    pass\n", encoding="utf-8")
    (root / "ara/tools/statistical_tester.py").write_text("def test_metric():\n    return True\n", encoding="utf-8")
    (root / "ara/templates_openml_curated_task01.py").write_text("DATASET = 'openml'\n", encoding="utf-8")
    (root / "ara/labs/bundle_ingest.py").write_text("SUPPORTED_BUNDLE_TYPES = {'aira_result_bundle'}\n", encoding="utf-8")
    (root / "scripts/revise_to_target_score.py").write_text("scientific_repair = True\n", encoding="utf-8")
    (root / "config.publishable.yaml").write_text(
        "research_domain: ai_ml\nrevision:\n  enable_scientific_repair: true\n",
        encoding="utf-8",
    )


def test_migration_inventory_tracks_production_runner_responsibilities(tmp_path):
    _write_fake_ara_source(tmp_path)

    payload = build_inventory(tmp_path)
    responsibilities = {item["id"]: item for item in payload["responsibilities"]}

    assert payload["source_exists"] is True
    assert payload["summary"]["present_responsibility_count"] == 7
    assert responsibilities["experiment_agent"]["aira_target"] == "aira.production_runner"
    assert responsibilities["experiment_agent"]["mvp_status"] == "ported_production_local"
    assert responsibilities["code_executor"]["aira_target"] == "aira.production_runner"
    assert responsibilities["code_executor"]["mvp_status"] == "ported_production_local"
    assert "aira.experiments run --profile production-local" in payload["summary"]["production_targets"]


def test_migration_inventory_cli_still_emits_json(tmp_path, capsys):
    _write_fake_ara_source(tmp_path)

    exit_code = cli.main(["migrate", "inventory", "--source", str(tmp_path), "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "passed"
    assert payload["summary"]["mvp_targets"] == [
        "research_lab.yaml",
        "aira.registries",
        "aira.bundles",
        "aira.run-fixture-benchmark",
    ]
    assert payload["summary"]["production_runner_status"] == "production_local_ported"
