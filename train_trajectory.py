"""
PTFN V7.3 - 临床动力学推演训练引擎 (The Clinical Forecasting Engine)
核心机制: 
1. 异构多任务 Loss (Poisson + BCE + Weighted BCE)
2. 潜状态惯性平滑正则化 (Smoothness Regularization)
3. 宗师级形态学底座冻结 (Foundation Pretraining Lockdown)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import os
import warnings

# 导入 V7.2 锁定的主架构
from dl_model import LatentDynamicsForecastingNet

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 启动 PTFN ICU 炼丹炉... [运算核心: {device}]")

def load_data(path):
    if not os.path.exists(path):
        exit(f"❌ 找不到数据文件 {path}，请先运行 build_dataset_factory.py")
    data = torch.load(path, map_location='cpu', weights_only=False)
    # 注意：对应 build_dataset_factory.py 中生成的标签
    return TensorDataset(data['X'], data['Y_pvc'], data['Y_afib'], data['Y_vt'])

def main():
    # 1. 准备数据管道
    print("⏳ 装载 ICU 临床时序切片数据...")
    train_dataset = load_data('dataset/ptfn_train.pt')
    val_dataset = load_data('dataset/ptfn_val.pt')
    
    # 冻结 CNN 后显存消耗大减，Batch Size 可以适当调大，加速训练
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 2. 实例化主模型
    model = LatentDynamicsForecastingNet().to(device)

    # =====================================================================
    # 🎯 V7.3 核心战术：预训练底座装载与冻结逻辑
    # =====================================================================
    backbone_path = 'models/ptbxl_backbone.pth'
    if os.path.exists(backbone_path):
        print("🌟 发现 PTB-XL 形态学宗师级权重！正在执行借尸还魂...")
        # strict=False 允许加载时忽略我们在预训练时加的临时分类头
        model.window_encoder.load_state_dict(torch.load(backbone_path, map_location=device, weights_only=True), strict=False)
        
        # 彻底冻结这双眼睛，切断它的梯度回传！
        for param in model.window_encoder.parameters():
            param.requires_grad = False
        print("🔒 底层 CNN 已彻底冻结。显存释放，GRU 算力全开，专注学习时序动力学！")
    else:
        print("⚠️ 警告：未发现 ptbxl_backbone.pth，模型将进行艰难的冷启动端到端训练。")
    # =====================================================================

    # 【极其重要】：优化器必须过滤掉被冻结的参数，否则会报错
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # 3. 异构损失函数群 (Heterogeneous Loss Landscape)
    poisson_loss = nn.PoissonNLLLoss(log_input=True)     # 处理 PVC 这种重尾离散计数过程
    bce_logits_loss = nn.BCEWithLogitsLoss()             # 处理 AFib 这种 [0,1] 的状态占比
    # 处理极其稀缺的 VT 发作，施加 20 倍正样本惩罚权重，逼迫模型宁可错杀不可放过
    vt_bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0]).to(device))

    LAMBDA_SMOOTH = 0.05
    EPOCHS = 50
    best_val_auprc = 0.0
    os.makedirs('models', exist_ok=True)

    for epoch in range(EPOCHS):
        # ------------------- 训练阶段 -------------------
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for bx, by_pvc, by_afib, by_vt in pbar:
            bx = bx.to(device)
            by_pvc = by_pvc.to(device)
            by_afib = by_afib.to(device)
            by_vt = by_vt.to(device)
            
            # 我们当前的 factory 生成的是未来 5 分钟的标签，所以视界强制设为 2 (5m)
            h_idx_5m = torch.full((bx.size(0),), 2, dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            out = model(bx, h_idx_5m)
            preds = out["preds"]
            
            # 计算异构 Loss
            loss_pvc = poisson_loss(preds["pvc_log_rate"], by_pvc)
            loss_afib = bce_logits_loss(preds["afib_logits"], by_afib)
            loss_vt = vt_bce_loss(preds["vt_hazard_logits"], by_vt)
            
            # 潜状态生物学惯性平滑正则化 (防止 GRU 潜变量出现非生理性的瞬间闪跳)
            h_seq = out["h_seq"] 
            diff_squared = (h_seq[:, 1:, :] - h_seq[:, :-1, :]) ** 2
            loss_smooth = torch.mean(torch.sum(diff_squared, dim=-1))
            
            # 临床风险加权联合 Loss (VT 风险占比 60%)
            total_loss = 0.1 * loss_pvc + 0.3 * loss_afib + 0.6 * loss_vt + LAMBDA_SMOOTH * loss_smooth
            total_loss.backward()
            
            # 梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            running_loss += total_loss.item()
            pbar.set_postfix(L_VT=f"{loss_vt.item():.2f}", Smth=f"{loss_smooth.item():.3f}")
            
        # ------------------- 验证阶段 -------------------
        model.eval()
        val_preds_vt = []
        val_true_vt = []
        
        with torch.no_grad():
            for bx, _, _, by_vt in val_loader:
                bx = bx.to(device)
                h_idx_5m = torch.full((bx.size(0),), 2, dtype=torch.long).to(device)
                out = model(bx, h_idx_5m)
                
                probs = torch.sigmoid(out["preds"]["vt_hazard_logits"]).cpu().numpy()
                val_preds_vt.extend(probs)
                val_true_vt.extend(by_vt.numpy())
                
        try:
            val_auprc = average_precision_score(val_true_vt, val_preds_vt)
        except:
            val_auprc = 0.0
            
        print(f"👉 Epoch {epoch+1} 总结 | Train Loss: {running_loss/len(train_loader):.4f} | Val VT AUPRC: {val_auprc:.4f}")
        
        # 永远保存最新 epoch 的权重，以便断点查看
        torch.save(model.state_dict(), 'models/ptfn_epoch_latest.pth')
        
        # 如果 AUPRC 破纪录，保存为最优权重
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            torch.save(model.state_dict(), 'models/ptfn_best.pth')
            print(f"🏆 新纪录！保存当前最优动力学权重至 models/ptfn_best.pth")

    print(f"🎉 训练大功告成！最优验证集 AUPRC: {best_val_auprc:.4f}")

if __name__ == "__main__":
    main()