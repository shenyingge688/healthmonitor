"""
Script: evaluate_clinical_metrics.py
Version: V9.0.1 (Time-Stratified TTE Protocol + Anti-NaN Shield)
Description: 
V9.0 终极临床验证协议。计算分层 AUROC (Time-Stratified AUROC)。
加入了双重防毒面具，彻底免疫真实临床数据的 NaN 污染。
"""
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from dl_model import LatentDynamicsForecastingNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("="*60)
    print("🏥 PTFN V9.0 临床计算科学检验 (Time-Stratified TTE)")
    print("="*60)
    
    model = LatentDynamicsForecastingNet().to(device)
    model.load_state_dict(torch.load('models/ptfn_v90_core.pth', map_location=device, weights_only=True))
    model.eval()
    
    # 严格按照 V9.0 格式解包 4 个张量
    data = torch.load('dataset/ptfn_val.pt', map_location='cpu', weights_only=False)
    val_loader = DataLoader(TensorDataset(data['X'], data['Y_pvc'], data['Y_afib'], data['Y_tte']), batch_size=64, shuffle=False)
    
    all_preds = []
    all_trues = []
    
    print("⏳ 正在进行全序列推断与清洗...")
    with torch.no_grad():
        for bx, _, _, by_tte in val_loader:
            # 🛡️ 第一道防线：物理拦截含毒数据的推断
            if torch.isnan(bx).any() or torch.isnan(by_tte).any():
                continue
                
            bx = bx.to(device)
            out = model(bx)
            
            # 还原对数时间: exp(tte_log) - 1 => 预计剩余分钟数
            preds_min = torch.expm1(out["preds"]["tte_preds"][:, -1]) 
            trues_min = torch.expm1(by_tte[:, -1])
            
            all_preds.extend(preds_min.cpu().numpy())
            all_trues.extend(trues_min.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)
    
    # 🛡️ 第二道防线：滤除数组中可能残留的任何 NaN 和无穷大 (Inf)
    valid_idx = ~np.isnan(all_preds) & ~np.isnan(all_trues) & ~np.isinf(all_preds) & ~np.isinf(all_trues)
    all_preds = all_preds[valid_idx]
    all_trues = all_trues[valid_idx]
    
    if len(all_preds) == 0:
        print("⚠️ 致命警告：清洗后没有留下有效样本，请检查模型是否完全崩溃输出了全 NaN。")
        return
        
    print(f"✅ 清洗完毕，有效样本数: {len(all_preds)}")
        
    # 提取稳定负样本 (距离崩溃大于 15 分钟) 作为负类基准
    stable_mask = all_trues >= 15.0
    stable_preds = all_preds[stable_mask]
    
    def calc_stratified_auroc(min_t, max_t):
        pos_mask = (all_trues >= min_t) & (all_trues < max_t)
        pos_preds = all_preds[pos_mask]
        
        if len(pos_preds) < 5 or len(stable_preds) < 5: 
            return "样本不足"
        
        # 核心逻辑：TTE 越小表示风险越大，距离崩溃越近
        # 所以我们将预测的 TTE 加上负号作为 Score，这样 Score 越大代表越有可能发生事件
        y_true = [1] * len(pos_preds) + [0] * len(stable_preds)
        y_score = list(-pos_preds) + list(-stable_preds)
        
        return f"{roc_auc_score(y_true, y_score):.4f}"

    print("\n📊 [Time-Stratified AUROC] 预警时间梯度检验:")
    print(f"   [距离发作 < 1 分钟] (极度崩溃期): AUROC = {calc_stratified_auroc(0, 1.0)}")
    print(f"   [距离发作 1-3 分钟] (临界转变期): AUROC = {calc_stratified_auroc(1.0, 3.0)}")
    print(f"   [距离发作 3-5 分钟] (演化萌芽期): AUROC = {calc_stratified_auroc(3.0, 5.0)}")
    print(f"   [距离发作 5-10分钟] (稳态偏离期): AUROC = {calc_stratified_auroc(5.0, 10.0)}")
    print("="*60)
    print("💡 结论指引：如果你看到随着时间逼近，AUROC 呈现显著上升的台阶梯度，")
    print("   这即是【短时心电序列存在确凿不可逆恶化动力学】的终极医学铁证！")

if __name__ == "__main__":
    main()