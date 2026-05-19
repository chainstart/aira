# 决策：AIRA 使用动态 spec 维护体系

日期：2026-05-19

状态：接受

## 背景

AIRA 当前是从 ARA 拆出的 AI/ML 领域实验室，已有 manifest、bundle validator、migration inventory 和 fixture benchmark。后续它会逐步承载真实 AI 实验环境、模型、数据集和 benchmark，因此需要能长期维护 requirement、任务状态和证据。

## 决策

AIRA 采用动态 spec 维护体系：

- `docs/aira_spec_governance.zh.md` 保存当前主 spec。
- `.engineering/spec_tasks.yaml` 保存机器可读任务台账。
- `docs/aira_implementation_status.zh.md` 保存人工状态矩阵。
- `docs/decisions/` 保存架构决策。
- `docs/spec_update_log.jsonl` 保存动态任务更新日志。

## 影响

- 真实 benchmark 迁入前，fixture smoke 和真实实验声明必须分开。
- ARA 只能基于 AIRA bundle 中声明的证据和复现说明使用结果。
- engineering-harness 开发 AIRA 任务后，应回写 spec 任务状态。
