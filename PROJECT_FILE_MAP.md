# ECG心律失常健康预警系统文件图

本项目正式能力限定为未来5分钟整体心律失常风险预警。AFib为外部研究支持的辅助方向，VT/VF与AT/SVT保留为探索性能力边界。

```text
C:\HealthMonitor
├─ README.md / requirements.txt / start_demo.ps1
├─ main.py / dashboard.py
├─ healthmonitor\              核心模型、信号、策略、回放、API和前端模块
├─ scripts\                    data、training、evaluation、external_validation
├─ tests\                      策略、三模型服务、冻结回放验收
├─ data\                       原始公开数据库（精简提交包不包含）
├─ datasets\trajectory_official\ 正式轨迹数据目录，可由环境变量覆盖
├─ models\official_ensemble\   正式三模型权重
├─ models\pretrained\          预训练权重
├─ artifacts\                  demo、evidence、policies、figures、explainability
├─ docs\                       报告与项目说明书
└─ submission_bundle\          待提交草稿包
```

## 关键入口

- `start_demo.ps1`：启动冻结资产前端演示。
- `python -m uvicorn healthmonitor.main:app --host 127.0.0.1 --port 8000`：可选实时API核验。
- `python tests/test_monitoring_policy.py`：正式报警策略测试。
- `python tests/test_ensemble_serving.py`：三模型服务概率回归。
- `python tests/test_demo_replay_consistency.py`：冻结病例/API一致性验收。

## 指标来源

所有报告和前端核心数字从 `artifacts/metric_registry.json` 读取。证据更新后必须重新运行：

```powershell
$env:PYTHONPATH='C:\HealthMonitor'
python scripts\evaluation\normalize_demo_manifest.py
python scripts\evaluation\build_metric_registry.py
```

## 提交边界

精简提交包包含源码、正式三模型、冻结演示资产、关键证据、报告、说明书、启动脚本和依赖文件；不包含原始公开数据库、训练张量、历史实验、旧模型和失败模型。

## 报告规则

- 当前报告固定为根目录 `HealthMonitor_ECG_Arrhythmia_Competition_Report.docx`。
- 不改变原格式、原目录、原排版，只做内容修订。
- 每轮历史版本保存在 `docs/report_history/`，索引见 `docs/report_history/INDEX.md`。
