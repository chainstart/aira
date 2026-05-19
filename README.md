# AIRA

AIRA is the AI/ML domain lab for the ARA ecosystem. This repository currently
provides a deterministic bootstrap contract:

- `python3 -m aira labs inspect --json` inspects `research_lab.yaml`.
- `python3 -m aira bundles validate <bundle> --json` validates an
  `aira_result_bundle`.
- `python3 -m aira migrate inventory --source <ara-repo> --json` inventories the
  AI experiment responsibilities that move out of legacy ARA.
- `python3 -m aira run-fixture-benchmark --out <dir> --json` emits a local,
  deterministic benchmark bundle with no live model calls.

The initial registries are placeholders for datasets, models, and benchmarks.
They are intentionally local and fixture-backed until real AI experiment runners
are migrated behind the same bundle contract.

## Spec Governance

AIRA tracks its implementation status through a dynamic spec maintenance system:

- `docs/aira_spec_governance.zh.md` describes the current AIRA requirements.
- `docs/aira_implementation_status.zh.md` records implementation status and gaps.
- `.engineering/spec_tasks.yaml` is the machine-readable task ledger used by
  Engineering Harness.
- `docs/decisions/` stores architecture decisions.

After each harness task or development stage, update the task ledger with the
implemented requirement, evidence command, artifact path, and remaining blocker.
