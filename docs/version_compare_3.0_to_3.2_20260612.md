# 3.0 与 3.2 版本差异对比报告

生成时间：2026-06-12 20:05
对比基线：本地 `3.0` 分支 -> 当前 `3.2` 分支待同步状态。
统计说明：重点比较源码、工程结构、证据材料、前端演示、报告归档和提交包；原始数据库、训练张量和模型权重原目录按 `.gitignore` 不作为普通源码差异展开。

## 一、总体结论

3.0 版本仍是以根目录脚本为主的早期 ECG 多分类/诊断型工程，主要由 `main.py`、`dashboard.py`、`dl_model.py`、`train_massive_v5.py`、`validation_multiclass.py` 等脚本支撑。

3.2 版本已经收敛为提交前工程形态：正式能力限定为“未来 5 分钟整体心律失常风险预警”，核心代码迁移到 `healthmonitor/` 包，训练、评估、外部验证和文档生成脚本迁移到 `scripts/`，冻结演示资产和证据材料进入 `artifacts/`，并新增 `tests/` 自动验收和 `submission_bundle/` 精简提交包。

核心变化不是继续训练新模型，而是完成项目定位、证据链、演示稳定性、报告内容和提交材料的集中整理。

## 二、文件层面变化

当前暂存区相对 3.0 的主要变化为：

- 新增约 224 个文件。
- 修改约 4 个文件。
- 删除约 7 个旧入口或缓存类文件。
- 大量旧 `current_core_v7` 内容迁移到 `healthmonitor/` 与 `scripts/`。

代表性新增内容：

- `healthmonitor/`：核心模型、信号处理、API、报警策略、路径解析和前端模块。
- `scripts/`：数据构建、训练、评估、外部验证、指标注册表和报告更新脚本。
- `tests/`：报警策略、集成推理、冻结回放一致性和提交物验收测试。
- `artifacts/`：冻结演示、验证证据、策略配置、图源和解释性材料。
- `docs/progress_history/`：每轮过程汇报归档。
- `docs/report_history/`：报告历史版本索引和草稿材料归档。
- `docs/version_compare_3.0_to_3.2_20260612.md`：本对比报告。
- `submission_bundle/HealthMonitor_PreSubmission_Draft_20260612.zip`：当前待提交草稿包。

## 三、工程结构变化

3.0 的工程形态较扁平，训练、推理、前端、验证脚本混在根目录，历史结果和协作记录也更容易与正式材料混杂。

3.2 的结构更接近可提交项目：

```text
C:\HealthMonitor
├─ README.md / requirements.txt / start_demo.ps1
├─ main.py / dashboard.py
├─ healthmonitor\
├─ scripts\
├─ tests\
├─ artifacts\
├─ docs\
└─ submission_bundle\
```

其中 `main.py` 和 `dashboard.py` 保留为根目录入口，实际实现由 `healthmonitor.main` 和 `healthmonitor.dashboard` 承担；路径解析统一由 `healthmonitor.paths` 管理，避免继续依赖 `current_core_v7` 或硬编码工作区路径。

## 四、能力边界变化

| 维度 | 3.0 版本 | 3.2 版本 |
|---|---|---|
| 项目定位 | ECG 多分类/诊断型原型 | 未来 5 分钟整体心律失常风险预警系统 |
| 正式能力 | 类别识别和早期风险输出并存 | 仅限定为整体心律失常预警 |
| 报警策略 | 口径未完全冻结 | `overall-risk-v1`，EWMA alpha 0.65，阈值 0.10，连续 2 点触发 |
| 模型服务 | 根目录脚本式服务 | `healthmonitor.main` FastAPI，三模型 logits 集成 |
| AFib | 内部/先导表述较多 | LTAFDB 83 记录主分析、84 记录敏感性分析支持的辅助研究方向 |
| VT/VF、AT/SVT | 容易被误读为待宣传能力 | 明确为探索性边界，不接入正式报警 |

## 五、前端演示变化

3.2 前端整理为三页式 Streamlit 演示：

- `监测预警`：首屏展示正式报警状态、整体风险、信号质量、模型投票数和风险离散度。
- `精选病例`：固定展示 MIT-BIH 119、100、201 三个精选病例，并保留可选实时 API 核验。
- `验证证据`：展示锁定测试、AFib 外部研究支持和能力边界。

固定演示点为：

- MIT-BIH 119 第 4 个推理点：PVC 正式报警，整体风险约 0.881，3/3 投票，信号质量 100%。
- MIT-BIH 100 第 17 个推理点：正常无报警，未来 Normal 约 0.988，整体风险约 0.012。
- MIT-BIH 201 第 10 个推理点：AFib 辅助方向，AFib 方向分数约 0.924，未来 AFib 概率约 0.573。

此外，本轮修复了 Streamlit 版本兼容风险：旧版 Streamlit 不支持 `st.tabs(default=...)` 和新版 `st.dataframe(width="stretch")` 参数，3.2 已加入兼容封装。

## 六、证据链变化

3.2 新增统一指标注册表 `artifacts/metric_registry.json`，前端和报告核心数字统一从注册表读取，避免手工重复填写。

新增或整理的证据包括：

- 锁定验证观察指标。
- 事件级报警提前量和误报 episode。
- 三模型集成验证输出。
- LTAFDB 全库 AFib 外部研究支持。
- 长尾类别边界审计。
- 软标签一致性评估。
- PVC 解释性样例。
- 冻结病例 replay manifest 与逐点 JSON。

这使 3.2 的材料从“实验过程可追溯”升级为“提交证据可复核”。

## 七、报告与历史资料变化

正式实验报告仍保留在根目录原位置，遵循“原格式、原目录、原排版，只改内容”的规则。3.2 内容更新去掉了正式报告中的内部轮次叙事、本地路径、`current_core_v7`、`V7/V8`、seed/checkpoint 等不适合提交的表述。

每轮过程汇报没有删除，已归档到：

- `docs/progress_history/`

报告历史版本和重建草稿归档到：

- `docs/report_history/`

注意：根目录 DOCX 受 `.gitignore` 约束，不作为独立 Git 文件跟踪；当前报告通过提交包 zip 进入 3.2 分支。

## 八、测试与验收变化

3.2 新增和强化了自动验收：

- `tests/test_monitoring_policy.py`
- `tests/test_ensemble_serving.py`
- `tests/test_demo_replay_consistency.py`
- `tests/test_submission_acceptance.py`

本轮已运行通过：

- 提交物验收。
- 冻结回放与 API 概率一致性，最大误差为 0。
- 正式报警策略测试。
- 三模型集成推理回归。
- 浏览器实际打开前端，默认病例和三页标签正常，无 Streamlit 报错。

## 九、提交包变化

3.2 新增：

- `submission_bundle/HealthMonitor_PreSubmission_Draft_20260612.zip`

该包包含源码、正式三模型、冻结演示资产、关键证据、报告、说明材料、启动脚本和依赖文件；不包含完整原始公开数据库和训练张量。由于封面占位符仍需最终补齐，当前包定位为“待提交草稿”。

提交包 zip 约 69 MB，低于 GitHub 单文件 100 MB 限制，但会显著增加仓库体积。

## 十、风险与注意事项

- 原始数据库、训练张量、模型权重原目录、根目录 DOCX 报告仍按 `.gitignore` 排除。
- 提交包 zip 内包含模型权重和报告副本，因此 GitHub 上可获得提交草稿包，但不是最终无占位符提交版。
- 正式报告封面中的作品 ID、学生类型和赛道仍需最终人工补齐。
- 后续若继续修改报告，应继续保留原版式，只做内容更新，并把过程汇报归档到 `docs/progress_history/`。

## 十一、建议

3.2 可以作为提交前稳定分支使用。下一步建议只做最终提交必要收敛：补齐封面占位符、确认最终 DOCX/PDF、重新跑四项验收并记录最终 SHA-256。除非另开实验分支，不建议在 3.2 上继续做长尾类别或新模型训练。
