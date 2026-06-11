# Project File Map

本文件记录 `C:\HealthMonitor` 的当前整理建议与主线边界。当前项目的唯一主线工作区是 `current_core_v7\`，后续所有新实验优先在这里展开。

## 推荐目标结构

```text
C:\HealthMonitor
├─ current_core_v7\              当前唯一主线工作区
├─ data\                         原始数据库，不动
├─ dataset_v7\                   当前正式 V7 数据集，不动
├─ models_v7_stage1\             当前正式模型，不动
├─ final_artifacts\              最终报告、最终策略、关键图表
├─ archive_experiments\          旧实验、旧模型、旧结果
├─ archive_legacy_root\          根目录旧脚本副本
└─ cleanup_candidates\           等确认后可删的临时文件
```

## 绝对不要删

- `C:\HealthMonitor\data\`
- `C:\HealthMonitor\dataset_v7\`
- `C:\HealthMonitor\models_v7_stage1\`
- `C:\HealthMonitor\current_core_v7\`
- `C:\HealthMonitor\current_core_v7\results_v7_round5\`
- `C:\HealthMonitor\current_core_v7\results_v7_round6\`
- `C:\HealthMonitor\current_core_v7\results_current\`
- `C:\HealthMonitor\HealthMonitor_ECG_Arrhythmia_Competition_Report.docx`
- `C:\HealthMonitor\参赛作品报告模版.docx`

特别说明：`current_core_v7\` 下的 `data`、`dataset_v7`、`models_v7_stage1`、`results` 等 junction/link 目录不要直接删除，以免误伤真实数据或模型。

## 当前主线文件

以下是当前主线优先保留的文件/脚本：

- `build_dataset_factory.py`
- `plan_dataset_v7.py`
- `audit_dataset_shards.py`
- `dl_model.py`
- `train_trajectory.py`
- `run_multiseed.py`
- `final_validation_analysis.py`
- `evaluate_smoothed_alert_policy.py`
- `evaluate_alert_policy.py`
- `evaluate_checkpoint_ensemble.py`
- `bootstrap_alert_policy_ci.py`
- `evaluate_adaptive_alert_policy.py`
- `evaluate_rr_gating.py`
- `calibrate_ensemble_uncertainty.py`
- `eval_ecg_level_attribution.py`
- `main.py`
- `dashboard.py`
- `test_ensemble_serving.py`
- `test_demo_samples.py`
- `README_CURRENT_CORE.md`
- `requirements.txt`

## 可归档，不建议直接删

- 根目录下的一批旧 `.py` 脚本副本
- `healthmonitor-3.0\`
- `.claude\`、`conversation-f3d00ba6\`、`history\`
- `models\`
- `results\`
- `current_core_v7\results_v7_round2\`
- `current_core_v7\results_v7_round3\`
- `current_core_v7\results_v7_round4\`
- `current_core_v7\models_v7_round2_*`
- `current_core_v7\models_v7_round4_*`
- `current_core_v7\models_v7_round5_h180_stage1\`
- `current_core_v7\dataset_v7_h180\`
- `build_ptbxl_factory.py`
- `pretrain_cnn_ptbxl.py`

## 可以删除的候选

先确认再删，优先级从高到低如下：

- `__pycache__\`
- 无用运行日志
- `~$*.docx` 这类 Word 临时锁文件
- 明确重复且不再引用的旧版报告
- 一次性 smoke/test 输出
- 已确认不用的临时脚本或临时结果

## 后续执行顺序

1. 先把 `current_core_v7\` 作为唯一开发入口。
2. 先归档，不急删历史模型和历史结果。
3. 只立即清理缓存、锁文件和确认无用的临时文件。
4. 每完成一轮优化后，同步更新：
   - `HealthMonitor_ECG_Arrhythmia_Competition_Report.docx`
   - `current_core_v7\results_v7_round6\ROUND6_PROGRESS.md`
   - 关键策略 JSON 和图表说明

## 推荐工作入口

```powershell
cd C:\HealthMonitor\current_core_v7
```
