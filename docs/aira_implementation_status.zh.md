# AIRA Spec 实现状态矩阵

日期：2026-05-19

状态：动态维护中

主 spec：`docs/aira_spec_governance.zh.md`

机器台账：`.engineering/spec_tasks.yaml`

决策记录：`docs/decisions/`

## Requirement 状态

| Requirement ID | 状态 | 当前证据 | 主要缺口 | 下一步 |
| --- | --- | --- | --- | --- |
| `REQ-AIRA-GOV-001` | `completed` | 本文档、`.engineering/spec_tasks.yaml`、决策记录 | 需在后续任务中持续更新 | 接入 harness spec-sync |
| `REQ-AIRA-MANIFEST-001` | `completed` | `research_lab.yaml`、`python3 -m aira labs inspect --json` | 随 ARA lab contract 演进 | 保持 manifest smoke |
| `REQ-AIRA-REGISTRY-001` | `partial` | fixture registry + local deterministic dataset/model/benchmark/ablation entries | 外部数据集、模型版本策略和资源生命周期仍不足 | 迁移真实 AI 实验资源 |
| `REQ-AIRA-BENCHMARK-001` | `completed` | `python3 -m aira run-local-benchmark --out <dir> --json`、local result bundle、provenance、ablation report、error analysis、run ledger | 仅覆盖小型本地确定性 benchmark，不含 GPU/外部数据/在线模型 | 增加更多本地 runner 和实验族 |
| `REQ-AIRA-BUNDLE-001` | `completed` | bundle validator、fixture result bundle | 需要更多实验类型 schema | 扩展 result bundle schema |
| `REQ-AIRA-AGENT-001` | `completed` | `python3 -m aira agent smoke --out <dir> --json`、agent plan/trace/observation/reflection artifacts、ablation/error-analysis observation | 仅执行已注册的本地确定性 benchmark | 增加更多本地 runner 后扩展 agent 选择策略 |
| `REQ-AIRA-MEMORY-001` | `partial` | bundle-local `memory/run_ledger.jsonl`、`artifacts/run_ledger_entry.json`、`memory/experiment_memory.json[l]`、`memory/agent_memory.json[l]` | 缺跨 run 的共享 memory service、失败重试和 agent 检索接口 | 后续建设共享本地 memory index |
| `REQ-AIRA-ARA-001` | `completed` | ARA handoff-ready agent bundle、`artifacts/ara_handoff.json`、`artifacts/reproducibility_notes.md`、bundle validator 的 `ara_gate` profile | 仅证明本地确定性 bundle 可被 ARA public gate 消费；未接入 ARA 仓库端到端测试 | ARA contract 变更时同步更新 handoff profile |

## 维护流程

1. 开发前检查 `.engineering/spec_tasks.yaml`，确认任务对应 requirement。
2. 开发后记录测试命令、benchmark bundle、registry 变更或 blocker。
3. 真实模型或外部数据依赖必须记录 reproducibility notes。
4. 未完成实验不得被 ARA 当作已复现结果使用。

## Agent MVP

`python3 -m aira agent smoke --out <dir> --json` 是当前稳定的本地 agent smoke。它执行最小 `plan -> act -> observe -> reflect` loop：

- `plan`：从 registry 选择 `local-text-outcome-classification`、`local-experiment-outcomes-v1` 和确定性 keyword/pass-prior/negative-term-ablation model。
- `act`：调用现有 `python3 -m aira run-local-benchmark` 等价 runner，生成基础 AIRA result bundle。
- `observe`：验证 bundle，记录 metrics、ablation/error-analysis summary、artifact ids 和 validation 状态。
- `reflect`：写入 `artifacts/agent_reflection.json`、`memory/experiment_memory.json[l]`、`memory/agent_memory.json` 和 `memory/agent_memory.jsonl`。

该 MVP 不访问网络、不调用在线模型、不依赖 GPU 或外部数据集。共享 memory service、跨 run 检索和多 runner 策略仍属于后续工作。

## ARA Handoff

`python3 -m aira agent smoke --out <dir> --json` 现在会在 bundle 中写入 ARA-facing handoff metadata：

- `artifacts/ara_handoff.json`：声明 `ara-public-bundle-reproduction-gate.v1` profile、bundle schema、验证命令、复现命令、必需 gate 输入、determinism flags 和 claims gate。
- `artifacts/reproducibility_notes.md`：记录本地复现命令、输入 fingerprints、无网络/无 GPU/无 live model API 限制和 agent smoke 复现说明。
- `memory/run_ledger.jsonl` 与 `artifacts/run_ledger_entry.json`：保留可机器读取的 bundle-local run ledger。
- `artifacts/ablation_report.json` 与 `artifacts/error_analysis.json`：记录确定性 negative-term ablation 及其错误类型。
- `memory/experiment_memory.json` 与 `memory/experiment_memory.jsonl`：保留可检索的本地实验记忆、fingerprints 和 ablation findings。

`python3 -m aira bundles validate <bundle> --json` 会在 validation metadata 中暴露 `ara_gate`，并检查 handoff metadata、reproducibility notes、claims、provenance 与 run ledger 是否能满足本地 ARA public reproduction gate。
