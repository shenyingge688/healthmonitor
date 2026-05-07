"""
HealthMonitor V3.0 - 性能测试与混淆矩阵校验脚本
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
from dl_model import HybridWarningNet
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📊 正在启动环境模型评估盲测程序... [运算设备: {device}]")

model = HybridWarningNet().to(device)
try:
    model.load_state_dict(torch.load('models/hybrid_v5_massive_best.pth', map_location=device, weights_only=True))
    model.eval()
except Exception as e:
    exit(f"❌ 权重加载环节失败：{e}")

try:
    data = torch.load('dataset/test_pure_v5.pt', map_location='cpu', weights_only=False)
    test_dataset = TensorDataset(data['X'], data['Y'])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"📦 验证样本载入量: {len(test_dataset)} 条数据")
except Exception as e:
    exit(f"❌ 张量文件装载过程失效: {e}")

all_preds, all_targets = [], []
start_time = time.time()

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x.to(device))
        all_preds.extend(np.argmax(outputs["prob"].cpu().numpy(), axis=1))
        all_targets.extend(np.argmax(batch_y.numpy(), axis=1))

all_preds, all_targets = np.array(all_preds), np.array(all_targets)
duration = time.time() - start_time

CLASS_NAMES = ["Class 0 (Normal)", "Class 1 (PVC)", "Class 2 (AFib)", "Class 3 (VF)", "Class 4 (VT/VFl)", "Class 5 (AT)"]
labels_idx = [0, 1, 2, 3, 4, 5]

clf_report = classification_report(all_targets, all_preds, labels=labels_idx, target_names=CLASS_NAMES, zero_division=0)
conf_matrix = confusion_matrix(all_targets, all_preds, labels=labels_idx)

# 【修复】使用 len(test_loader) (批次总数) 代替 len(all_preds) (样本总数)
true_batch_time_ms = duration / max(1, len(test_loader)) * 1000

print(f"""
======================================================
🏥 预警引擎框架预测综合性能呈现
======================================================
* 平均时延核算: {true_batch_time_ms:.3f} ms/每批次 (Batch Size: 32)
------------------------------------------------------
【详细指标呈现表】
{clf_report}

【分类关联对角线散布阵列】
{conf_matrix}
======================================================
""")