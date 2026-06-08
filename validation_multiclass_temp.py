import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from dl_model import HybridWarningNet
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import time

# 1. [ok]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[ok] [ok] [[ok]-6[ok]] [ok]... [[ok]: {device}]")

# 2. [ok]
model = HybridWarningNet().to(device)
try:
    # [ok]
    model.load_state_dict(torch.load('models/hybrid_v5_massive_best.pth', map_location=device, weights_only=True))
    model.eval()
    print("[ok] V3.0 [ok]")
except Exception as e:
    print(f"[ok] [ok]{e}")
    exit()

# 3. [ok]
try:
    data = torch.load('dataset/train_massive_v5.pt', map_location='cpu', weights_only=False)
    X, Y = data['X'], data['Y']
    
    # [ok] 30% [ok]
    dataset = TensorDataset(X, Y)
    test_size = int(0.3 * len(dataset))
    _, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - test_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"[ok] [ok]: {len(test_dataset)} [ok]")
except Exception as e:
    print(f"[ok] [ok]: {e}")
    exit()

# 4. [ok]
all_preds = []
all_targets = []
start_time = time.time()

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        
        # [ok] 6 [ok] (Batch, 6)
        probs = outputs["prob"].cpu().numpy()
        # [ok] Top-1 [ok]
        preds = np.argmax(probs, axis=1) 
        
        # [ok] Label Smoothing [ok] (Batch, 6) [ok]
        targets = np.argmax(batch_y.numpy(), axis=1) 
        
        all_preds.extend(preds)
        all_targets.extend(targets)

# 5. [ok]
all_preds = np.array(all_preds)
all_targets = np.array(all_targets)
duration = time.time() - start_time

CLASS_NAMES = ["Class 0 (Normal)", "Class 1 (PVC)", "Class 2 (AFib)", "Class 3 (VF)", "Class 4 (VT/VFl)", "Class 5 (AT)"]

# [ok]
# [ok] 0-5 [ok]
labels_idx = [0, 1, 2, 3, 4, 5]
clf_report = classification_report(all_targets, all_preds, labels=labels_idx, target_names=CLASS_NAMES, zero_division=0)
conf_matrix = confusion_matrix(all_targets, all_preds, labels=labels_idx)

report = f"""
======================================================
[ok] (HybridWarningNet V3.0) [ok]
======================================================
[ok]CrossEntropyLoss + [ok] [cite: 213, 267]
------------------------------------------------------
[ok]
[ok] Class 3 ([ok] VF) [ok] Class 4 ([ok] VT) [ok] Recall ([ok]/[ok]) [ok]
[ok] 0.95 [ok]

{clf_report}

[ok]
* [ok]: {duration/len(all_preds)*1000:.3f} ms/frame
* [ok]
{conf_matrix}
======================================================
"""
print(report)