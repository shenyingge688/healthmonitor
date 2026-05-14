"""
Script: evaluate_clinical_metrics.py
Version: V8.3 Final (Clinical Computational Science Protocol)
"""
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score
from dl_model import LatentDynamicsForecastingNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_risk(logits):
    hazards = 0.5 * torch.sigmoid(logits)
    log_survival = torch.sum(torch.log(1.0 - hazards + 1e-6), dim=1)
    return 1.0 - torch.exp(log_survival)

def extract_causal_predictions(model, loader, mode='baseline'):
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for bx, _, _, by_vt in loader:
            bx = bx.to(device)
            if mode == 'shuffle':
                bx = bx[:, torch.randperm(bx.size(1)), :, :]
            elif mode == 'reverse':
                bx = torch.flip(bx, dims=[1])
                
            out = model(bx)
            final_logits = out["preds"]["vt_hazard_logits"][:, -1, :]
            
            all_preds.extend(compute_risk(final_logits).cpu().numpy())
            is_event = (by_vt[:, -1, :] == 1).any(dim=1).float()
            all_trues.extend(is_event.numpy())
    return np.array(all_preds), np.array(all_trues)

def evaluate_latent_matched_control(model, loader, epsilon=5.0):
    model.eval()
    all_z, all_labels, all_preds = [], [], []
    print("🔬 执行 Latent-Matched 轨迹匹配扫描...")
    with torch.no_grad():
        for bx, _, _, by_vt in loader:
            bx = bx.to(device)
            
            B, S, C, L = bx.shape
            x_flat = bx.view(B * S, C, L)
            z_flat = model.window_encoder(x_flat)
            z_seq = z_flat.view(B, S, -1)
            z_global = torch.mean(z_seq, dim=1) 
            
            out = model(bx)
            risk = compute_risk(out["preds"]["vt_hazard_logits"][:, -1, :])
            is_event = (by_vt[:, -1, :] == 1).any(dim=1).float()
            
            all_z.append(z_global)
            all_preds.append(risk)
            all_labels.append(is_event)
            
    Z = torch.cat(all_z, dim=0).cpu()
    Y = torch.cat(all_labels, dim=0).cpu()
    P = torch.cat(all_preds, dim=0).cpu()
    
    vt_idx = torch.where(Y == 1)[0]
    ctrl_idx = torch.where(Y == 0)[0]
    
    valid_pairs = []
    for v_i in vt_idx:
        dists = torch.cdist(Z[v_i].unsqueeze(0), Z[ctrl_idx])[0]
        min_dist, min_idx = torch.min(dists), torch.argmin(dists)
        
        if min_dist < epsilon:
            valid_pairs.append({'vt_risk': P[v_i].item(), 'ctrl_risk': P[ctrl_idx[min_idx]].item()})
            
    if len(valid_pairs) > 0:
        vt_risks = [p['vt_risk'] for p in valid_pairs]
        ctrl_risks = [p['ctrl_risk'] for p in valid_pairs]
        pair_y = [1]*len(vt_risks) + [0]*len(ctrl_risks)
        pair_preds = vt_risks + ctrl_risks
        auroc = roc_auc_score(pair_y, pair_preds)
        print(f"🎯 成功找到 {len(valid_pairs)} 对形态学双胞胎！动力学独立剥离 AUROC: {auroc:.4f}")
    else:
        print("⚠️ 未找到满足严苛条件的双胞胎样本。")

def main():
    print("="*60)
    print("🏥 PTFN 计算临床科学检验平台 (V8.3 Causal Protocol)")
    print("="*60)
    
    model = LatentDynamicsForecastingNet().to(device)
    model.load_state_dict(torch.load('models/ptfn_v83_core.pth', map_location=device, weights_only=True))
    
    data = torch.load('dataset/ptfn_val.pt', map_location='cpu', weights_only=False)
    val_loader = DataLoader(TensorDataset(data['X'], data['Y_pvc'], data['Y_afib'], data['Y_vt']), batch_size=64, shuffle=False)
    
    base_p, trues = extract_causal_predictions(model, val_loader, mode='baseline')
    base_auprc = average_precision_score(trues, base_p)
    print(f"[1] 自然因果流 AUPRC:      {base_auprc:.4f}")
    
    shuf_p, _ = extract_causal_predictions(model, val_loader, mode='shuffle')
    shuf_auprc = average_precision_score(trues, shuf_p)
    print(f"[2] 拓扑打乱后 AUPRC:      {shuf_auprc:.4f} (Drop: {base_auprc - shuf_auprc:.4f})")
    
    rev_p, _ = extract_causal_predictions(model, val_loader, mode='reverse')
    rev_auprc = average_precision_score(trues, rev_p)
    print(f"[3] 时光倒流后 AUPRC:      {rev_auprc:.4f} (Drop: {base_auprc - rev_auprc:.4f})")
    
    evaluate_latent_matched_control(model, val_loader)

if __name__ == "__main__":
    main()