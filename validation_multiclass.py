"""
PTFN - 临床多视界推演验证引擎 (Validation)
功能: 在严格的时间轴物理隔离测试集上，评估模型在 30秒、1分钟、5分钟 三个独立视界下的预测性能。
"""
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import time
import os
import warnings

# 【修复】导入全新的 PTFN 网络架构
from dl_model import LatentDynamicsForecastingNet

# 屏蔽 PyTorch 未来版本警告
warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📊 启动 PTFN 多视界全景评估程序... [运算设备: {device}]")

# ==========================================
# 1. 引擎装载
# ==========================================
model = LatentDynamicsForecastingNet().to(device)

# 读取你在 train_trajectory.py 中最新保存的权重
model_path = 'models/ptfn_epoch_latest.pth' 
try:
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"✅ 成功加载推演权重: {model_path}")
except Exception as e:
    exit(f"❌ 权重加载失败，请检查模型路径。错误: {e}")

# ==========================================
# 2. 数据装载 (读取严格物理隔离的验证集)
# ==========================================
data_path = 'dataset/ptfn_val.pt'
try:
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    val_dataset = TensorDataset(data['X'], data['Y_30s'], data['Y_1m'], data['Y_5m'])
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    print(f"📦 验证集载入成功: 共 {len(val_dataset)} 条隔离长程轨迹")
except Exception as e:
    exit(f"❌ 数据加载失败，请检查数据路径。错误: {e}")

# ==========================================
# 3. 推演与数据收集
# ==========================================
all_preds_30s, all_targets_30s = [], []
all_preds_1m,  all_targets_1m  = [], []
all_preds_5m,  all_targets_5m  = [], []

start_time = time.time()

with torch.no_grad():
    for bx, by_30s, by_1m, by_5m in val_loader:
        outputs = model(bx.to(device))
        
        # 收集 30 秒视界预测
        preds_30s = np.argmax(outputs["prob_30s"].cpu().numpy(), axis=1)
        all_preds_30s.extend(preds_30s)
        all_targets_30s.extend(by_30s.numpy())
        
        # 收集 1 分钟视界预测
        preds_1m = np.argmax(outputs["prob_1m"].cpu().numpy(), axis=1)
        all_preds_1m.extend(preds_1m)
        all_targets_1m.extend(by_1m.numpy())
        
        # 收集 5 分钟视界预测
        preds_5m = np.argmax(outputs["prob_5m"].cpu().numpy(), axis=1)
        all_preds_5m.extend(preds_5m)
        all_targets_5m.extend(by_5m.numpy())

duration = time.time() - start_time

CLASS_NAMES = ["0 (Normal)", "1 (PVC)", "2 (AFib)", "3 (VF)", "4 (VT/VFl)", "5 (AT)"]
labels_idx = [0, 1, 2, 3, 4, 5]

print(f"""
======================================================
🏥 PTFN 临床生理轨迹推演系统 - 性能透视报告
======================================================
* 平均时延核算: {duration / max(1, len(val_loader)) * 1000:.3f} ms/每批次
* 评估数据源: {data_path} (已应用 Time-Block 物理隔离)
------------------------------------------------------
""")

print("【一、 超短期预警视界 (+30s Forecasting)】")
print("-> 评估模型对 R-on-T 等即刻爆发风险的抓取能力")
print(classification_report(all_targets_30s, all_preds_30s, labels=labels_idx, target_names=CLASS_NAMES, zero_division=0))

print("\n------------------------------------------------------")
print("【二、 短期预警视界 (+1m Forecasting)】")
print("-> 评估模型对频发早搏、短阵室速等中期前兆的推演")
print(classification_report(all_targets_1m, all_preds_1m, labels=labels_idx, target_names=CLASS_NAMES, zero_division=0))

print("\n------------------------------------------------------")
print("【三、 中长期预警视界 (+5m Forecasting)】")
print("-> 评估模型对病理演化 (如 HRV 崩溃、持续失稳) 的长程推演")
print(classification_report(all_targets_5m, all_preds_5m, labels=labels_idx, target_names=CLASS_NAMES, zero_division=0))

print("======================================================")