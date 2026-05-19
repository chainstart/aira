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
| `REQ-AIRA-REGISTRY-001` | `partial` | fixture registry + local deterministic dataset/model/benchmark entries | 外部数据集、模型版本策略和资源生命周期仍不足 | 迁移真实 AI 实验资源 |
| `REQ-AIRA-BENCHMARK-001` | `completed` | `python3 -m aira run-local-benchmark --out <dir> --json`、local result bundle、provenance、run ledger | 仅覆盖小型本地确定性 benchmark，不含 GPU/外部数据/在线模型 | 后续在 agent MVP 中接入更多 runner |
| `REQ-AIRA-BUNDLE-001` | `completed` | bundle validator、fixture result bundle | 需要更多实验类型 schema | 扩展 result bundle schema |
| `REQ-AIRA-AGENT-001` | `pending` | 暂无完整 agent loop | 缺 plan-act-observe-reflect 和工具注册表 | 设计 AI 实验 agent MVP |
| `REQ-AIRA-MEMORY-001` | `partial` | bundle-local `memory/run_ledger.jsonl` 和 `artifacts/run_ledger_entry.json` | 缺跨 run 的共享 memory service、失败重试和 agent 检索接口 | 在 AIRA agent MVP 中扩展 memory schema |
| `REQ-AIRA-ARA-001` | `partial` | migration inventory 和 bundle contract | 需要跨仓库消费 smoke 和复现 gate | 增加 ARA + AIRA integration test |

## 维护流程

1. 开发前检查 `.engineering/spec_tasks.yaml`，确认任务对应 requirement。
2. 开发后记录测试命令、benchmark bundle、registry 变更或 blocker。
3. 真实模型或外部数据依赖必须记录 reproducibility notes。
4. 未完成实验不得被 ARA 当作已复现结果使用。
