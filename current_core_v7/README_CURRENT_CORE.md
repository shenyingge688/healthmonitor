# Current Core V7

这个目录是当前第二轮优化的主线工作区。根目录原文件没有被删除；这里保留一份当前核心脚本副本，并用 Windows junction 链接到原始数据、模型和结果，避免重复复制大文件。

## 当前主线代码

- `build_dataset_factory.py`: 从原始 ECG 数据构建轨迹样本，包含 10 分钟历史、5 分钟未来软标签、RR 特征和 transition weight。
- `plan_dataset_v7.py`: 规划 V7 patient-wise train/val/test/demo split。
- `audit_dataset_shards.py`: 只读审计 V7 shard 的形状、标签分布、软标签有效性和非有限值。
- `train_trajectory.py`: 当前训练入口，支持 `--dataset-dir`、`--output-root`、多 checkpoint 选择和 risk-aware 辅助损失参数。
- `run_multiseed.py`: 多随机种子训练入口。
- `final_validation_analysis.py`: 当前正式评估入口，包含 discrimination、AUPRC、calibration、bootstrap CI、locked threshold、event lead-time 等。
- `evaluate_alert_policy.py`: 实时报警策略评估，支持验证集选策略、测试集锁定应用。
- `dl_model.py`: ArrhythmiaWarningNet 模型结构。
- `constants.py`, `ecg_utils.py`: 应用和信号处理通用配置/工具。
- `dashboard.py`, `main.py`: 演示/API 相关入口。

## 链接的大文件目录

- `data/` -> `C:\HealthMonitor\data`: 原始 ECG 数据库。
- `dataset_v7/` -> `C:\HealthMonitor\dataset_v7`: 当前 V7 train/val/test shard。
- `models/` -> `C:\HealthMonitor\models`: 旧 V6 模型、多种子结果和 PTB-XL backbone。
- `models_v7_stage1/` -> `C:\HealthMonitor\models_v7_stage1`: 当前 V7 stage1 正式模型。
- `results/` -> `C:\HealthMonitor\results`: 全部评估结果。

## 当前正式结果快捷入口

`results_current/` 只放当前主线最重要的结果链接：

- `dataset_v7_plan/`: V7 数据划分方案。
- `dataset_v7_audit/`: V7 数据审计。
- `v7_stage1_val/`: V7 stage1 在验证集上的正式评估。
- `v7_stage1_test_locked/`: V7 stage1 在测试集上的 locked evaluation。
- `alert_val_lockedbasis/`: 验证集报警策略选择。
- `alert_test_applied_valpolicy/`: 测试集只应用验证集策略后的报警评估。

## 推荐工作方式

后续第二轮优化优先在这个目录运行命令，例如：

```powershell
cd C:\HealthMonitor\current_core_v7
python train_trajectory.py --dataset-dir dataset_v7 --output-root models_v7_round2 --epochs 20 --seed 0
```

正式评估仍建议遵守：验证集选择 calibration/threshold/policy，测试集只应用锁定配置。
