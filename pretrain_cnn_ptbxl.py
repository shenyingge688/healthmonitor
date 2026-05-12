"""
Phase 1 - PTB-XL CNN 骨干网络预训练引擎
仅训练 WindowEncoder，让其成为形态学特征提取大师
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
import warnings

# 导入你 V7.2 架构里现成的这双“眼睛”
from dl_model import WindowEncoder 

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"👁️ 启动 PTB-XL 形态学宗师预训练中心... [运算核心: {device}]")

# 包装一个临时用于 5 分类的外壳
class PretrainClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = WindowEncoder(in_channels=1)
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5) # 输出 5 个超级类
        )
        
    def forward(self, x):
        feat = self.encoder(x)
        return self.head(feat)

def main():
    if not os.path.exists('dataset/ptbxl_train.pt'):
        exit("❌ 找不到数据集，请先运行 build_ptbxl_factory.py")

    print("📦 载入两万份心电图切片...")
    train_data = torch.load('dataset/ptbxl_train.pt', map_location='cpu', weights_only=True)
    val_data = torch.load('dataset/ptbxl_val.pt', map_location='cpu', weights_only=True)

    # 数据量大，Batch Size 可以开到 128 或 256
    train_loader = DataLoader(TensorDataset(train_data['X'], train_data['Y']), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data['X'], val_data['Y']), batch_size=128, shuffle=False)

    model = PretrainClassifier().to(device)
    
    # 多标签分类必须使用 BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # 当验证集指标不再上升时，自动将学习率减半
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    EPOCHS = 20
    best_auc = 0.0
    os.makedirs('models', exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for bx, by in pbar:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            # 加入梯度裁剪护城河，把无限大的梯度强行锁死在 1.0 以内！
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix(Loss=f"{loss.item():.4f}")
            
        # 验证环节
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                logits = model(bx.to(device))
                all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                all_targets.extend(by.numpy())
                
        # 计算 5 大类的平均 AUROC (Macro AUROC)
        try:
            val_auc = roc_auc_score(all_targets, all_preds, average='macro')
        except:
            val_auc = 0.0
            
        print(f"👉 总结 | Train Loss: {running_loss/len(train_loader):.4f} | Val Macro AUROC: {val_auc:.4f}")
        
        scheduler.step(val_auc)
        
        if val_auc > best_auc:
            best_auc = val_auc
            # 【核心剥离操作】：只保存 encoder，把临时分类头扔进垃圾桶！
            torch.save(model.encoder.state_dict(), 'models/ptbxl_backbone.pth')
            print(f"🌟 新纪录！已成功剥离并保存最优底层权重至 models/ptbxl_backbone.pth")

    print(f"🎉 预训练大功告成，PTB-XL 最优 AUROC: {best_auc:.4f}")

if __name__ == "__main__":
    main()