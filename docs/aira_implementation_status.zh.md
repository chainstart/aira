# AIRA Spec 实现状态矩阵

日期：2026-05-20

状态：动态维护中

主 spec：`docs/aira_spec_governance.zh.md`

机器台账：`.engineering/spec_tasks.yaml`

决策记录：`docs/decisions/`

## Requirement 状态

| Requirement ID | 状态 | 当前证据 | 主要缺口 | 下一步 |
| --- | --- | --- | --- | --- |
| `REQ-AIRA-GOV-001` | `completed` | 本文档、`.engineering/spec_tasks.yaml`、决策记录 | 需在后续任务中持续更新 | 接入 harness spec-sync |
| `REQ-AIRA-MANIFEST-001` | `completed` | `research_lab.yaml`、`python3 -m aira labs inspect --json` | 随 ARA lab contract 演进 | 保持 manifest smoke |
| `REQ-AIRA-REGISTRY-001` | `completed` | fixture/local registry、production-local dataset/model/benchmark entries、`python3 -m aira registry audit --profile production-local --json` | 生产 profile 下外部 adapter 仅登记为 disabled template，未启用下载或 live model | 后续在新 profile 中显式启用真实外部 adapter |
| `REQ-AIRA-BENCHMARK-001` | `completed` | `python3 -m aira run-local-benchmark --out <dir> --json`、local result bundle、provenance、ablation report、error analysis、run ledger | 仅覆盖小型本地确定性 benchmark，不含 GPU/外部数据/在线模型 | 增加更多本地 runner 和实验族 |
| `REQ-AIRA-BUNDLE-001` | `completed` | bundle validator、fixture result bundle | 需要更多实验类型 schema | 扩展 result bundle schema |
| `REQ-AIRA-AGENT-001` | `completed` | `python3 -m aira agent smoke --out <dir> --json`、agent plan/trace/observation/reflection artifacts、ablation/error-analysis observation | 仅执行已注册的本地确定性 benchmark | 增加更多本地 runner 后扩展 agent 选择策略 |
| `REQ-AIRA-MEMORY-001` | `completed` | bundle-local memory artifacts、`python3 -m aira memory index --runs <bundle-or-parent> --out <dir> --json`、cross-run `memory_index.json`/`runs.jsonl`/`failures.jsonl`/fingerprint/outcome/reflection indexes | 当前是本地文件 index，不是远程共享 service；自动过期、失败重试执行和主动 agent planning 仍属后续能力 | 扩展 agent 使用 memory index 选择 production-local run |
| `REQ-AIRA-ARA-001` | `completed` | ARA handoff-ready agent bundle、`artifacts/ara_handoff.json`、`artifacts/reproducibility_notes.md`、bundle validator 的 `ara_gate` profile | 仅证明本地确定性 bundle 可被 ARA public gate 消费；未接入 ARA 仓库端到端测试 | ARA contract 变更时同步更新 handoff profile |
| `REQ-AIRA-PROD-RUNNER-001` | `completed` | `python3 -m aira experiments run --profile production-local --plan tests/fixtures/production_plan.json --out <dir> --json`、production-local bundle、policy report、execution trace、run ledger | 仅支持显式 `production-local` profile、本地 inline Python、无 package install/网络/GPU/live model API；容器级隔离仍属外部运行环境 | 后续接入生产 registry/evaluation/memory |
| `REQ-AIRA-PROD-REGISTRY-001` | `completed` | production-local registry profile、local cache/operator-supplied/disabled optional external adapters、fingerprint/version/license/resource/reproducibility metadata、registry audit CLI | 未启用网络下载、GPU、live model API 或外部模型权重；operator artifact license 需由 operator attestation 提供 | `AIRA-PROD-EVAL-001` |
| `REQ-AIRA-PROD-EVAL-001` | `completed` | `python3 -m aira experiments evaluate --bundle <bundle> --json`、production evaluation metrics、ablation matrix、error taxonomy、exact paired statistical tests、report summary artifacts | 当前 evaluator 只消费 bundle 内已物化的 prediction CSV；ablation 是确定性 failure-keyword 禁用对比，不训练或调用 live model；小 fixture 的 exact test 统计功效有限 | `AIRA-PROD-MEMORY-001` |
| `REQ-AIRA-PROD-MEMORY-001` | `completed` | `python3 -m aira memory index --runs /tmp/aira_prod_runner_bundle --out /tmp/aira_prod_memory --json`、失败 ledger、fingerprint index、dataset/model outcome matrix、agent reflection retrieval | lifecycle 目前是 rebuild/status/max-run filter；未提供远程 service、自动 expiry 或自动 retry execution | `AIRA-PROD-ARA-001` |
| `REQ-AIRA-PROD-ARA-001` | `completed` | `python3 -m aira agent production-smoke --out <dir> --json`、`python3 -m aira bundles validate <bundle> --profile ara-production --json`、`research_lab.yaml` 的 `ara-production` handoff profile | 仍为 production-local smoke；不启用网络、GPU、外部数据集、package install 或 live model API | ARA contract 变更时同步更新 production handoff profile |

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

## 生产级 AI 实验迁移

生产级迁移用于承接原 ARA 旧 AI/ML 实验执行能力，不改变默认 smoke 的确定性边界。执行顺序交给 `engineering-harness`：

1. `AIRA-PROD-RUNNER-001`：已迁移旧 ARA 实验执行安全边界，形成 AIRA `production-local` controlled runner。
2. `AIRA-PROD-REGISTRY-001`：已扩展 dataset/model/benchmark registry，支持 local cache、operator-supplied artifacts、disabled optional external adapter contract、fingerprint、versioning、license/resource policy 和 reproducibility notes。
3. `AIRA-PROD-EVAL-001`：已迁移 statistical testing、ablation、error analysis 和报告 artifacts。
4. `AIRA-PROD-MEMORY-001`：已建立跨 run experiment memory、失败 ledger、fingerprint/outcome index 与 agent reflection 检索。
5. `AIRA-PROD-ARA-001`：发布 ARA production-local handoff profile，使 ARA 只通过 manifest 和 bundle contract 消费生产实验输出。

## Production ARA Handoff

`python3 -m aira agent production-smoke --out <dir> --json` 会生成 production-local ARA handoff bundle：

- 先执行 `production-local` runner，再追加 production evaluation 和 bundle-local memory index。
- 写入 `artifacts/ara_handoff.json`，声明 `ara-production` validation profile、`research_lab.yaml` dispatch、`aira_result_bundle` bundle type、runner/evaluator/memory artifacts 和必需 gate inputs。
- 写入 `artifacts/reproducibility_notes.md`，记录 production smoke 复现命令、`ara-production` 验证命令、fingerprints 和无网络/无 GPU/无 live model 限制。
- `python3 -m aira bundles validate <bundle> --profile ara-production --json` 会额外检查 production runner/evaluation、policy report、execution trace、task summary、production evaluation reports、run ledger、memory index 和 dispatch metadata。

该 handoff 不要求 ARA 导入 AIRA 内部模块；ARA 侧只需 manifest dispatch 与 bundle validation/result ingestion。

## ARA Handoff

`python3 -m aira agent smoke --out <dir> --json` 现在会在 bundle 中写入 ARA-facing handoff metadata：

- `artifacts/ara_handoff.json`：声明 `ara-public-bundle-reproduction-gate.v1` profile、bundle schema、验证命令、复现命令、必需 gate 输入、determinism flags 和 claims gate。
- `artifacts/reproducibility_notes.md`：记录本地复现命令、输入 fingerprints、无网络/无 GPU/无 live model API 限制和 agent smoke 复现说明。
- `memory/run_ledger.jsonl` 与 `artifacts/run_ledger_entry.json`：保留可机器读取的 bundle-local run ledger。
- `artifacts/ablation_report.json` 与 `artifacts/error_analysis.json`：记录确定性 negative-term ablation 及其错误类型。
- `memory/experiment_memory.json` 与 `memory/experiment_memory.jsonl`：保留可检索的本地实验记忆、fingerprints 和 ablation findings。

`python3 -m aira bundles validate <bundle> --json` 会在 validation metadata 中暴露 `ara_gate`，并检查 handoff metadata、reproducibility notes、claims、provenance 与 run ledger 是否能满足本地 ARA public reproduction gate。

## Production-Local Runner

`python3 -m aira experiments run --profile production-local --plan <plan.json> --out <dir> --json` 是当前生产级迁移的显式入口。它承接旧 ARA `ExperimentAgent`/`CodeExecutor` 中适合迁入 AIRA 的职责：

- profile gate：只接受 `production-local`，默认 smoke/local benchmark 不会隐式进入生产 runner。
- policy checks：校验 plan schema、task graph、依赖环、package allowlist、危险字符串、危险 import、输出路径和 deterministic flags。
- resource bounds：每个 task 使用 subprocess timeout、CPU thread 环境变量和 stdout/stderr 截断。
- failure isolation：task 在 `work/tasks/<task_id>` 下独立执行；失败 task 的 dependent task 会被标记为 skipped。
- artifact materialization：声明输出被复制到 `artifacts/tasks/<task_id>/`，并写入 `artifact_manifest.json`、`artifacts/policy_report.json`、`artifacts/execution_trace.json`、`artifacts/task_summary.json`、`artifacts/provenance.json`、`artifacts/reproduction_status.json`、`artifacts/run_ledger_entry.json` 和 `memory/run_ledger.jsonl`。

## Production Evaluation

`python3 -m aira experiments evaluate --bundle <bundle> --json` 会读取 production-local bundle 中已物化的 prediction CSV，并把评估结果追加回同一个 bundle：

- `artifacts/production_evaluation/metrics.json`：primary metrics、preferred-label baseline、per-class metrics 和 confusion matrix。
- `artifacts/production_evaluation/ablation_matrix.json`：确定性 failure-keyword ablation matrix、delta metrics 和 changed predictions。
- `artifacts/production_evaluation/error_taxonomy.json`：primary/ablation error rows 与 error type counts。
- `artifacts/production_evaluation/statistical_tests.json`：primary vs ablation 的 exact McNemar paired test 和 effect sizes。
- `artifacts/production_evaluation/report_summary.json`：机器可读 summary、artifact paths 和后续建议。

该命令会更新 `artifact_manifest.json`、`claims.json` 和 `bundle_manifest.json`，并再次执行 bundle validation。它不访问网络、不安装 package、不调用 live model，也不重新运行实验 task。

## Production Memory

`python3 -m aira memory index --runs <bundle-or-parent> --out <dir> --json` 会把 bundle-local memory 提升为本地跨 run index：

- `memory_index.json`：汇总 run summaries、failure ledger、fingerprints、dataset/model outcomes 和 retrieval keys。
- `runs.jsonl`：每个 run 的 ledger、metrics、provenance fingerprints、evaluation summary、experiment memory 和 agent memory。
- `failures.jsonl`：run/task failure、production evaluation error taxonomy 和 ablation regression 记录。
- `fingerprints.json`：按 run 和按 fingerprint 值反查 dataset/model/registry input fingerprints。
- `outcomes.json`：按 dataset/model 聚合通过/失败次数和 best accuracy。
- `reflections.json`：从 `artifacts/agent_reflection.json` 或 `memory/agent_memory.json[l]` 提取可检索 reusable memory。

该 index 支持 `--status`、`--max-runs` 和 `--keep-existing` lifecycle 控制；它仍是本地文件 index，不提供远程 service、自动过期或自动重试执行。
