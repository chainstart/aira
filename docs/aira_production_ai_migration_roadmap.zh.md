# AIRA 生产级 AI 实验迁移路线图

日期：2026-05-21

本文把“从 ARA 迁出旧 AI/ML 实验执行能力”拆成可由 `engineering-harness` 实施的任务包。当前 AIRA 已具备本地确定性 benchmark/agent MVP；生产级能力仍未完成。

## 迁移目标

- 将旧 ARA 的实验 runner、安全执行边界、统计检验、ablation、error analysis、dataset/model 配置和实验记忆迁入 AIRA。
- 保持默认 smoke/local benchmark 无网络、无 GPU、无 live model API，适合 CI 和 ARA public gate。
- 生产级能力通过显式 `production-local` profile 开启，并在 result bundle 中记录资源、fingerprint、失败和复现说明。

## Harness 任务包

| 顺序 | 任务 | Requirement | 状态 | 验收重点 |
| --- | --- | --- | --- | --- |
| 1 | `AIRA-PROD-RUNNER-001` | `REQ-AIRA-PROD-RUNNER-001` | completed | `production-local` 受控 runner、策略校验、资源限制、失败隔离、artifact materialization |
| 2 | `AIRA-PROD-REGISTRY-001` | `REQ-AIRA-PROD-REGISTRY-001` | completed | local cache、operator-supplied artifact、可选外部 adapter、fingerprint、license/resource policy |
| 3 | `AIRA-PROD-EVAL-001` | `REQ-AIRA-PROD-EVAL-001` | completed | metrics、ablation matrix、error taxonomy、statistical tests、机器可读报告 |
| 4 | `AIRA-PROD-MEMORY-001` | `REQ-AIRA-PROD-MEMORY-001` | completed | 本地跨 run index、失败台账、dataset/model outcome 检索、agent reflection |
| 5 | `AIRA-PROD-ARA-001` | `REQ-AIRA-PROD-ARA-001` | completed | `research_lab.yaml` production-local/open dispatch、`ara-production` / `ara-production-open` bundle validation、ARA handoff notes |
| 6 | `AIRA-ARA-DEEPENING-001` | `REQ-AIRA-ARA-DEEPENING-001` | completed | 消费 ARA deepening task，生成 follow-up production plan，输出可被 ARA rescoring 的 evidence-deepening bundle |

## 不回迁到 ARA 的内容

- 任意 AI/ML 实验脚本执行。
- 数据集下载、模型训练、GPU 或 live model API 调用。
- 实验 memory、ablation、error analysis、statistical testing 的领域内部逻辑。

ARA 只保留 manifest dispatch、bundle validation、claim/evidence/reproduction gate 和 dashboard/catalog 展示。

## AIRA-PROD-RUNNER-001 结果

已新增 `python3 -m aira experiments run --profile production-local --plan <plan> --out <bundle> --json`。该 runner 只执行显式声明的本地 inline Python task，禁止 package installation、网络、GPU、external datasets 和 live model API；输出 bundle 记录 policy report、execution trace、task summary、provenance、reproduction status、run ledger 和 materialized task artifacts。

## AIRA-PROD-EVAL-001 结果

已新增 `python3 -m aira experiments evaluate --bundle <bundle> --json`。该 evaluator 读取 production-local bundle 中的 prediction CSV，追加 `artifacts/production_evaluation/metrics.json`、`ablation_matrix.json`、`error_taxonomy.json`、`statistical_tests.json` 和 `report_summary.json`，并更新 artifact manifest、claims 与 bundle manifest 后重新验证 bundle。

当前实现保持完全本地和确定性：不重新执行实验、不训练模型、不访问网络、不调用 live model。统计检验为 primary prediction 与 deterministic failure-keyword ablation 的 exact paired test；小型 fixture 的显著性结论只用于 contract smoke，不应作为生产效果证据。

## AIRA-PROD-ARA-001 结果

已新增 `python3 -m aira agent production-smoke --out <bundle> --json`。该 smoke 会执行 `production-local` runner、追加 production evaluation、生成 bundle-local memory index，并写入 `artifacts/ara_handoff.json` 与 `artifacts/reproducibility_notes.md`。

`research_lab.yaml` 现在声明 `ara-production` handoff profile；ARA 只需要读取 `research_lab.yaml` 与 `aira_result_bundle`，再调用 `python3 -m aira bundles validate <bundle> --profile ara-production --json`。该 profile 会检查 production runner/evaluation、policy/trace/task summary、run ledger、production memory index、dispatch metadata 与必需 gate inputs。

## AIRA-ARA-DEEPENING-001 结果

已新增 `python3 -m aira experiments deepen --profile production-open --task <task.json> --source-bundle <bundle> --out <bundle> --json`，用于承接 ARA 顶会闭环的 evidence gap 补强任务。该入口会：

- 读取 ARA 的 `ara.research_deepening_task_package.v1`。
- 生成 `aira.ara_deepening_plan.v1` production plan。
- 调用既有 `production-open` runner 执行补强任务。
- 输出 `primary_contribution`、`mechanism_insight`、`artifact_availability`、`top_venue_evidence`、`deepening_report` 等 artifacts。
- 让 `claims.json` 引用上述 artifacts 和 reproduction/policy/trace/provenance/ledger artifacts，供 ARA 重新执行 claim/evidence scoring。

`research_lab.yaml` 的 `ara-production-open` profile 已从固定 fixture plan 改为 `experiments deepen`，并接收 ARA 传入的 `<task>`、`<source_bundle>` 和 `<bundle>`。这使 ARA 的 `top-venue improve-loop` 能把每轮失败的证据缺口真正交给 AIRA 生成 follow-up bundle，而不是重复执行固定 smoke plan。
