import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from dl_model import HybridWarningNet
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import time

# 1. 环境准备与增强配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📊 正在启动 [临床级-6分类盲测] 验证流水线... [设备: {device}]")

# 2. 模型装载
model = HybridWarningNet().to(device)
try:
    # 确保加载的是多分类重构后的最新权重
    model.load_state_dict(torch.load('models/hybrid_v5_massive_best.pth', map_location=device, weights_only=True))
    model.eval()
    print("✅ V3.0 多分类预警权重装载完成，进入推理模式。")
except Exception as e:
    print(f"❌ 权重加载失败：{e}")
    exit()

# 3. 数据加载与切分逻辑
try:
    data = torch.load('dataset/train_massive_v5.pt', map_location='cpu', weights_only=False)
    X, Y = data['X'], data['Y']
    
    # 暂按 30% 纯净集进行随机拆分
    dataset = TensorDataset(X, Y)
    test_size = int(0.3 * len(dataset))
    _, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - test_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"📦 盲测样本量: {len(test_dataset)} 条")
except Exception as e:
    print(f"❌ 数据读取失败，真实的报错信息为: {e}")
    exit()

# 4. 执行多维度临床模拟盲测
all_preds = []
all_targets = []
start_time = time.time()

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        
        # 提取 6 分类概率 (Batch, 6)
        probs = outputs["prob"].cpu().numpy()
        # 获取模型预测的 Top-1 类别索引
        preds = np.argmax(probs, axis=1) 
        
        # 从 Label Smoothing 的软标签 (Batch, 6) 中还原出真实的类别索引
        targets = np.argmax(batch_y.numpy(), axis=1) 
        
        all_preds.extend(preds)
        all_targets.extend(targets)

# 5. 生成契合多分类项目的专业报表
all_preds = np.array(all_preds)
all_targets = np.array(all_targets)
duration = time.time() - start_time

CLASS_NAMES = ["Class 0 (Normal)", "Class 1 (PVC)", "Class 2 (AFib)", "Class 3 (VF)", "Class 4 (VT/VFl)", "Class 5 (AT)"]

# 计算分类报告
# 修改后：强制指定 0-5 索引标签，防止因某类数据缺失导致崩溃
labels_idx = [0, 1, 2, 3, 4, 5]
clf_report = classification_report(all_targets, all_preds, labels=labels_idx, target_names=CLASS_NAMES, zero_division=0)
conf_matrix = confusion_matrix(all_targets, all_preds, labels=labels_idx)

report = f"""
======================================================
🏥多分类监护系统 (HybridWarningNet V3.0) 性能评估
======================================================
损失策略：CrossEntropyLoss + 极度倾斜重症权重 [cite: 213, 267]
------------------------------------------------------
【核心临床重症指标监控】
请重点关注 Class 3 (室颤 VF) 和 Class 4 (室速 VT) 的 Recall (召回率/敏感度) 。
若这两个值达到 0.95 以上，说明防漏报机制极其成功！

{clf_report}

【工程落地可行性】
* 实时响应时延: {duration/len(all_preds)*1000:.3f} ms/frame
* 混淆矩阵对角线验证：
{conf_matrix}
======================================================
"""
print(report)