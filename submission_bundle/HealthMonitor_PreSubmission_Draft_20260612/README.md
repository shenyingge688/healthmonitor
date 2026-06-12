# ECG心律失常健康预警系统

基于单导联ECG的未来5分钟整体心律失常风险预警系统。正式演示使用冻结病例资产，无需完整原始数据库即可启动前端；实时API核验为可选功能。

## 快速启动

```powershell
pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File .\start_demo.ps1
```

可选启动API：

```powershell
python -m uvicorn healthmonitor.main:app --host 127.0.0.1 --port 8000
```

## 验收测试

```powershell
$env:PYTHONPATH='C:\HealthMonitor'
python tests\test_monitoring_policy.py
python tests\test_ensemble_serving.py
python tests\test_demo_replay_consistency.py
```

## 能力边界

- 正式能力：未来5分钟整体心律失常风险预警。
- 辅助研究：AFib方向分数，来自LTAFDB 83记录主分析、84记录敏感性分析。
- 探索边界：VT/VF、AT/SVT不作为现场成功病例，不接入正式报警。

## 报告维护

当前竞赛报告保留在根目录 `HealthMonitor_ECG_Arrhythmia_Competition_Report.docx`。后续只在原文件、原目录、原排版基础上修改内容；历史版本保存在 `docs/report_history/`。
