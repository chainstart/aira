# AIRA Spec Governance

日期：2026-05-20

状态：初始动态 spec

## 1. 定位

AIRA 是 ARA 生态中的 AI/ML 领域实验室。它负责 AI 研究所需的数据集、模型、benchmark、实验运行器、结果包和复现实验环境。ARA 平台可以调用 AIRA，但不应把 AI 领域实验代码内嵌回 ARA。

## 2. 当前稳定接口

- `research_lab.yaml`：ARA-facing lab manifest。
- `python3 -m aira labs inspect --json`：检查 lab manifest。
- `python3 -m aira bundles validate <bundle> --json`：验证 `aira_result_bundle`。
- `python3 -m aira migrate inventory --source <ara-repo> --json`：盘点从 ARA 迁出的 AI 实验职责。
- `python3 -m aira run-fixture-benchmark --out <dir> --json`：生成确定性 fixture benchmark bundle。
- `python3 -m aira run-local-benchmark --out <dir> --json`：运行本地确定性文本分类 benchmark，生成含 provenance、ablation report、error analysis、run ledger 与 experiment memory 的 `aira_result_bundle`。
- `python3 -m aira agent smoke --out <dir> --json`：生成 ARA handoff-ready 的确定性 agent bundle，包含 `artifacts/ara_handoff.json`、`artifacts/reproducibility_notes.md`、claims、provenance、ablation/error-analysis artifacts 和 bundle-local memory。
- `python3 -m aira experiments run --profile production-local --plan <plan> --out <bundle> --json`：运行确定性 production-local 计划。
- `python3 -m aira experiments run --profile production-open --plan <plan> --out <bundle> --json`：运行显式 open production 计划，可下载、使用外部数据集、安装 package、使用 GPU 和 live model/API。
- `python3 -m aira registry audit --profile production-open --json`：审计 production-open dataset/model/benchmark registry surface。

## 3. Requirement ID

- `REQ-AIRA-GOV-001`：动态 spec 维护体系。
- `REQ-AIRA-MANIFEST-001`：ARA-compatible lab manifest 和 CLI。
- `REQ-AIRA-REGISTRY-001`：dataset/model/benchmark registry。
- `REQ-AIRA-BENCHMARK-001`：确定性 fixture benchmark 和真实 benchmark 迁移接口。
- `REQ-AIRA-BUNDLE-001`：`aira_result_bundle` contract。
- `REQ-AIRA-AGENT-001`：AI 实验自动化 agent loop。
- `REQ-AIRA-MEMORY-001`：实验记忆、运行台账和失败记录。
- `REQ-AIRA-ARA-001`：ARA-facing result bundle 和复现接口。
- `REQ-AIRA-PROD-RUNNER-001`：生产级 AI 实验 runner，承接旧 ARA `ExperimentAgent`/`CodeExecutor` 的安全边界、脚本执行、资源限制和 artifact materialization。
- `REQ-AIRA-PROD-REGISTRY-001`：生产级 dataset/model registry，支持外部数据源、local cache、fingerprint、版本策略和资源生命周期。
- `REQ-AIRA-PROD-EVAL-001`：生产级 evaluation、ablation、error analysis 和 statistical testing artifacts。
- `REQ-AIRA-PROD-MEMORY-001`：跨 run experiment memory、失败台账、检索接口和 agent 可复用经验。
- `REQ-AIRA-PROD-ARA-001`：ARA production handoff profile，使 ARA 通过 `research_lab.yaml` 与 `aira_result_bundle` 消费生产级实验结果。

## 4. 生产级迁移边界

原 ARA 中的 AI/ML 实验能力应迁入 AIRA，但必须分成两层：

- 默认层：本地确定性 smoke/local benchmark，无网络、无 GPU、无 live model API，作为 CI 和 ARA public reproduction gate 的稳定输入。
- 生产层：显式 profile。`production-local` 保持确定性本地执行；`production-open` 恢复旧 ARA 生产实验的开放能力，允许下载、外部数据集、package install、GPU 和 live model/API，并在 bundle 中记录资源、fingerprint、失败、统计检验和复现说明。

生产层迁移的最小完成标准：

1. AIRA 能在受控 runner 中执行已声明的实验计划，并 materialize 数据、模型、日志、metrics、ablation、error-analysis 和统计检验 artifacts。
2. Dataset/model registry 能区分 built-in fixture、local cache、外部下载和 operator-supplied artifact，并记录 fingerprint 与 license/资源策略。
3. Experiment memory 能跨 run 检索失败、模型表现、dataset fingerprint 和 agent reflection。
4. ARA 只能通过 manifest dispatch 与 result bundle 消费 AIRA production-local / production-open 输出，不能 import AIRA 内部实验代码。

## 5. 动态维护规则

1. 新增 AI 实验能力前，先写入 requirement ID 和任务台账。
2. 每个任务完成后，必须记录测试命令、fixture/local bundle、benchmark 结果或 blocker。
3. 真实模型调用、数据下载和昂贵 benchmark 必须和 fixture smoke 分离。
4. ARA 消费 AIRA 结果时，只能依据 bundle 中声明的 evidence 和 reproducibility notes。
5. engineering-harness 接手 AIRA 任务后，应在任务结束时更新 `.engineering/spec_tasks.yaml` 和 `docs/spec_update_log.jsonl`。
