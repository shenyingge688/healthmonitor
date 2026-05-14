"""
Script: pretrain_cnn_ptbxl.py
Phase 1 - PTB-XL CNN 骨干网络预训练引擎 (适配 V10.1 架构)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
import warnings

# 导入 V10 架构里全新的“眼睛”
from dl_model import WindowEncoder 

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"👁️ 启动 PTB-XL 形态学宗师预训练中心 (V10 适配版)... [运算核心: {device}]")

class PretrainClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # V10 的 WindowEncoder 默认输出 embed_dim=256
        self.encoder = WindowEncoder(in_channels=1, embed_dim=256)
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            # 🚀 核心修改：接收来自 V10 Encoder 的 256 维特征
            nn.Linear(256, 64), 
            nn.ReLU(),
            nn.Linear(64, 5) # 输出 PTB-XL 的 5 个超级类
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

    train_loader = DataLoader(TensorDataset(train_data['X'], train_data['Y']), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data['X'], val_data['Y']), batch_size=128, shuffle=False)

    model = PretrainClassifier().to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    EPOCHS = 20
    best_auc = 0.0
    os.makedirs('models', exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", dynamic_ncols=True)
        
        for bx, by in pbar:
            # 适配 V10 的 Float16/Float32 显存策略
            bx = bx.to(device, dtype=torch.float32) 
            by = by.to(device, dtype=torch.float32)
            
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix(Loss=f"{loss.item():.4f}")
            
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device, dtype=torch.float32)
                logits = model(bx)
                all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                all_targets.extend(by.numpy())
                
        try:
            val_auc = roc_auc_score(all_targets, all_preds, average='macro')
        except:
            val_auc = 0.0
            
        print(f"👉 总结 | Train Loss: {running_loss/len(train_loader):.4f} | Val Macro AUROC: {val_auc:.4f}")
        
        scheduler.step(val_auc)
        
        if val_auc > best_auc:
            best_auc = val_auc
            # 剥离 V10 版本的 Encoder 权重并保存
            torch.save(model.encoder.state_dict(), 'models/ptbxl_backbone.pth')
            print(f"🌟 新纪录！已成功剥离并保存最优底层权重至 models/ptbxl_backbone.pth")

    print(f"🎉 预训练大功告成，PTB-XL 最优 AUROC: {best_auc:.4f}")

if __name__ == "__main__":
    main()