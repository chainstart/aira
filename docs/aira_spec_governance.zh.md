# AIRA Spec Governance

日期：2026-05-19

状态：初始动态 spec

## 1. 定位

AIRA 是 ARA 生态中的 AI/ML 领域实验室。它负责 AI 研究所需的数据集、模型、benchmark、实验运行器、结果包和复现实验环境。ARA 平台可以调用 AIRA，但不应把 AI 领域实验代码内嵌回 ARA。

## 2. 当前稳定接口

- `research_lab.yaml`：ARA-facing lab manifest。
- `python3 -m aira labs inspect --json`：检查 lab manifest。
- `python3 -m aira bundles validate <bundle> --json`：验证 `aira_result_bundle`。
- `python3 -m aira migrate inventory --source <ara-repo> --json`：盘点从 ARA 迁出的 AI 实验职责。
- `python3 -m aira run-fixture-benchmark --out <dir> --json`：生成确定性 fixture benchmark bundle。
- `python3 -m aira run-local-benchmark --out <dir> --json`：运行本地确定性文本分类 benchmark，生成含 provenance 与 run ledger 的 `aira_result_bundle`。
- `python3 -m aira agent smoke --out <dir> --json`：生成 ARA handoff-ready 的确定性 agent bundle，包含 `artifacts/ara_handoff.json`、`artifacts/reproducibility_notes.md`、claims、provenance 和 bundle-local run ledger。

## 3. Requirement ID

- `REQ-AIRA-GOV-001`：动态 spec 维护体系。
- `REQ-AIRA-MANIFEST-001`：ARA-compatible lab manifest 和 CLI。
- `REQ-AIRA-REGISTRY-001`：dataset/model/benchmark registry。
- `REQ-AIRA-BENCHMARK-001`：确定性 fixture benchmark 和真实 benchmark 迁移接口。
- `REQ-AIRA-BUNDLE-001`：`aira_result_bundle` contract。
- `REQ-AIRA-AGENT-001`：AI 实验自动化 agent loop。
- `REQ-AIRA-MEMORY-001`：实验记忆、运行台账和失败记录。
- `REQ-AIRA-ARA-001`：ARA-facing result bundle 和复现接口。

## 4. 动态维护规则

1. 新增 AI 实验能力前，先写入 requirement ID 和任务台账。
2. 每个任务完成后，必须记录测试命令、fixture/local bundle、benchmark 结果或 blocker。
3. 真实模型调用、数据下载和昂贵 benchmark 必须和 fixture smoke 分离。
4. ARA 消费 AIRA 结果时，只能依据 bundle 中声明的 evidence 和 reproducibility notes。
5. engineering-harness 接手 AIRA 任务后，应在任务结束时更新 `.engineering/spec_tasks.yaml` 和 `docs/spec_update_log.jsonl`。
