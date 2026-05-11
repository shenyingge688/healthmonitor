"""
HealthMonitor V4.6 - 性能测试与混淆矩阵校验脚本 (双任务全景透视版)
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
from dl_model import HybridWarningNet
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📊 正在启动双通道全景评估程序... [运算设备: {device}]")

model = HybridWarningNet().to(device)
try:
    model.load_state_dict(torch.load('models/hybrid_v5_massive_best.pth', map_location=device, weights_only=True))
    model.eval()
except Exception as e:
    exit(f"❌ 权重加载失败：{e}")

try:
    data = torch.load('dataset/test_pure_v5.pt', map_location='cpu', weights_only=False)
    test_dataset = TensorDataset(data['X'], data['Y_future'], data['Y_current'], data['HRV'])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"📦 验证样本载入量: {len(test_dataset)} 条数据")
except Exception as e:
    exit(f"❌ 张量文件装载过程失效: {e}")

all_preds_fut, all_targets_fut = [], []
all_preds_cur, all_targets_cur = [], []
start_time = time.time()

with torch.no_grad():
    for batch_x, batch_y_future, batch_y_current, batch_hrv in test_loader:
        outputs = model(batch_x.to(device), hrv_features=batch_hrv.to(device))
        
        # 1. 收集 [未来预警] 的推断结果
        all_preds_fut.extend(np.argmax(outputs["prob"].cpu().numpy(), axis=1))
        all_targets_fut.extend(np.argmax(batch_y_future.numpy(), axis=1))
        
        # 2. 收集 [当下诊断] 的推断结果
        preds_cur = torch.softmax(outputs["logits_current"], dim=1)
        all_preds_cur.extend(np.argmax(preds_cur.cpu().numpy(), axis=1))
        all_targets_cur.extend(np.argmax(batch_y_current.numpy(), axis=1))

all_preds_fut, all_targets_fut = np.array(all_preds_fut), np.array(all_targets_fut)
all_preds_cur, all_targets_cur = np.array(all_preds_cur), np.array(all_targets_cur)
duration = time.time() - start_time

CLASS_NAMES = ["0 (Normal)", "1 (PVC)", "2 (AFib)", "3 (VF)", "4 (VT/VFl)", "5 (AT)"]
labels_idx = [0, 1, 2, 3, 4, 5]

print(f"""
======================================================
🏥 HealthMonitor V4.6 预警引擎框架双任务性能透视
======================================================
* 平均时延核算: {duration / max(1, len(test_loader)) * 1000:.3f} ms/每批次
------------------------------------------------------
""")

print("【一、 当下状态诊断能力 (Task: Current State)】")
print("-> 评估模型底层的波形特征识别能力是否正常")
print(classification_report(all_targets_cur, all_preds_cur, labels=labels_idx, target_names=CLASS_NAMES, zero_division=0))
print("对角线散布阵列:")
print(confusion_matrix(all_targets_cur, all_preds_cur, labels=labels_idx))

print("\n------------------------------------------------------")
print("【二、 超前 5 分钟预警能力 (Task: Future Forecasting)】")
print("-> 评估模型推断未来的能力")
print(classification_report(all_targets_fut, all_preds_fut, labels=labels_idx, target_names=CLASS_NAMES, zero_division=0))
print("对角线散布阵列:")
print(confusion_matrix(all_targets_fut, all_preds_fut, labels=labels_idx))
print("======================================================")