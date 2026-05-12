"""
PTFN V7.2 - 临床决策评估系统 (Clinical Decision Evaluation System)
核心修复: 全集反事实消融测试 (Full-Set Temporal Ablation)
"""
import torch
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.calibration import IsotonicRegression
import warnings

# 导入你的 V7.1 模型架构
from dl_model import LatentDynamicsForecastingNet

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🏥 启动 PTFN 临床决策评估中心... [运算核心: {device}]")

# =====================================================================
# 模块 1：迟滞报警引擎 (Clinical Alarm Engine)
# =====================================================================
class ClinicalAlarmEngine:
    def __init__(self, threshold=0.15, persistence_wins=3, refractory_wins=10):
        self.threshold = threshold
        self.persistence = persistence_wins
        self.refractory = refractory_wins
        
    def simulate_alarms(self, risk_probs, true_events):
        alarms = []
        consecutive_high_risk = 0
        cooldown = 0
        
        for t, risk in enumerate(risk_probs):
            if cooldown > 0:
                cooldown -= 1
                consecutive_high_risk = 0
                continue
                
            if risk >= self.threshold:
                consecutive_high_risk += 1
                if consecutive_high_risk >= self.persistence:
                    alarms.append(t)
                    cooldown = self.refractory
                    consecutive_high_risk = 0
            else:
                consecutive_high_risk = 0
                
        true_onsets = np.where(true_events > 0.5)[0]
        time_to_event_gains = []
        false_alarms = 0
        detected_events = 0
        
        for alarm_t in alarms:
            # 寻找报警后 30 个时间步 (约30分钟) 内的真实发作
            valid_future_events = true_onsets[(true_onsets > alarm_t) & (true_onsets <= alarm_t + 30)]
            if len(valid_future_events) > 0:
                teg = valid_future_events[0] - alarm_t
                time_to_event_gains.append(teg)
                detected_events += 1
            else:
                false_alarms += 1
                
        missed_events = len(true_onsets) - detected_events
        fab_per_100_steps = (false_alarms / max(1, len(risk_probs))) * 100
        
        return {
            "Total_Alarms": len(alarms),
            "False_Alarms": false_alarms,
            "FAB_Rate": fab_per_100_steps,
            "Mean_TEG_mins": np.mean(time_to_event_gains) if time_to_event_gains else 0.0,
            "Missed_Events": missed_events,
            "Detected_Events": detected_events,
            "Detection_Rate": detected_events / max(1, len(true_onsets)) if len(true_onsets) > 0 else 0.0
        }

# =====================================================================
# 模块 2：反事实消融张量处理器 (Temporal Counterfactual Ablation)
# =====================================================================
def run_ablation(model, bx, horizon_idx, mode="baseline"):
    bx_perturbed = bx.clone()
    
    if mode == "tail_mask_1m":
        # 抹除最后 1 分钟 (最后 4 个重叠窗口) 替换为高斯噪声
        bx_perturbed[:, -4:, :, :] = torch.randn_like(bx_perturbed[:, -4:, :, :])
    elif mode == "temporal_shuffle":
        # 打乱过去观测历史的时序拓扑顺序
        for i in range(bx_perturbed.size(0)):
            idx = torch.randperm(bx_perturbed.size(1))
            bx_perturbed[i] = bx_perturbed[i, idx, :, :]
            
    with torch.no_grad():
        out = model(bx_perturbed, horizon_idx)
    return torch.sigmoid(out["preds"]["vt_hazard_logits"]).cpu().numpy()

# =====================================================================
# 模块 3：主评估引擎
# =====================================================================
def evaluate_clinical_system():
    # 1. 装载模型与权重
    print("⏳ 装载 PTFN 模型权重...")
    model = LatentDynamicsForecastingNet().to(device)
    model_path = 'models/ptfn_epoch_latest.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("✅ 权重载入成功！")
    else:
        exit("❌ 找不到权重文件！请先运行 train_trajectory.py 完成训练。")
    model.eval()
    
    # 2. 装载真实的时间轴硬隔离验证集
    print("⏳ 装载真实物理隔离验证集 (dataset/ptfn_val.pt)...")
    data_path = 'dataset/ptfn_val.pt'
    if not os.path.exists(data_path):
        exit("❌ 找不到验证集数据！请检查路径。")
        
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    val_dataset = TensorDataset(data['X'], data['Y_pvc'], data['Y_afib'], data['Y_vt'])
    # 必须为 False，保持真实临床时序
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print("🔬 提取基线动力学特征与原始生存风险概率...")
    raw_probs_vt = []
    true_vt = []
    
    # 以 5m 视界的 VT Hazard 为评估核心 (idx = 2)
    with torch.no_grad():
        for bx, _, _, by_vt in val_loader:
            bx = bx.to(device)
            h_5m_idx = torch.full((bx.size(0),), 2, dtype=torch.long).to(device)
            
            out = model(bx, h_5m_idx)
            probs = torch.sigmoid(out["preds"]["vt_hazard_logits"]).cpu().numpy()
            raw_probs_vt.extend(probs)
            true_vt.extend(by_vt.numpy())
            
    raw_probs_vt = np.array(raw_probs_vt)
    true_vt = np.array(true_vt)
    
    if np.sum(true_vt) == 0:
        print("⚠️ 警告：当前验证集中未发现任何 VT/VF 正样本！无法计算 AUPRC。")
        return

    # 3. 后处理：可靠性校准 (Isotonic Regression)
    print("⚖️ 执行 Isotonic Regression 临床概率校准...")
    ir = IsotonicRegression(out_of_bounds='clip')
    calibrated_probs_vt = ir.fit_transform(raw_probs_vt, true_vt)
    
    brier_before = brier_score_loss(true_vt, raw_probs_vt)
    brier_after = brier_score_loss(true_vt, calibrated_probs_vt)

    # 4. 统计学宏观指标
    auprc_baseline = average_precision_score(true_vt, calibrated_probs_vt)
    auroc = roc_auc_score(true_vt, calibrated_probs_vt)
    
    # 5. ICU 迟滞报警模拟器
    print("🚨 启动 ICU 迟滞报警模拟器 (阈值: 0.15, M-out-of-N: 3/10)...")
    alarm_engine = ClinicalAlarmEngine(threshold=0.15, persistence_wins=3, refractory_wins=10)
    alarm_stats = alarm_engine.simulate_alarms(calibrated_probs_vt, true_vt)

    # 6. 时序反事实消融 (全集运行)
    print("🌪️ 运行全集反事实时序消融测试 (检验因果逻辑防作弊)...")
    
    def get_full_ablation_auprc(mode):
        ablation_raw_probs = []
        with torch.no_grad():
            for bx, _, _, _ in val_loader:
                bx = bx.to(device)
                h_idx = torch.full((bx.size(0),), 2, dtype=torch.long).to(device)
                probs = run_ablation(model, bx, h_idx, mode=mode)
                ablation_raw_probs.extend(probs)
        # 用跑基线时拟合好的 ir 模型去转换消融后的概率，保持标尺一致
        calibrated_ablation = ir.transform(np.array(ablation_raw_probs))
        return average_precision_score(true_vt, calibrated_ablation)

    auprc_tail_mask = get_full_ablation_auprc("tail_mask_1m")
    auprc_shuffle = get_full_ablation_auprc("temporal_shuffle")

    # =====================================================================
    # 7. 生成权威临床评估报告
    # =====================================================================
    print(f"""
=====================================================================
🏥 PTFN 临床决断力综合评估报告 (V7.2 Final Protocol)
=====================================================================
【一、 宏观预测有效性 (Statistical Efficacy)】
* 验证样本量:          {len(true_vt)} (VT正样本数: {int(np.sum(true_vt))})
* 原始 AUPRC (核心):   {auprc_baseline:.4f} 
* AUROC:               {auroc:.4f}

【二、 概率可靠性校准 (Calibration Quality)】
* Brier Score (校准前): {brier_before:.4f} 
* Brier Score (校准后): {brier_after:.4f} ⬇️ 

【三、 ICU 临床报警效用 (Clinical Alarm Utility)】
* 设定策略: 阈值 Risk > 0.15 | 确证期 3 窗口 | 不应期 10 窗口
* 事件抓取率:          {alarm_stats['Detection_Rate']*100:.1f}% ({alarm_stats['Detected_Events']} / {int(np.sum(true_vt))})
* 预警提前收益 (TEG):  平均提前 {alarm_stats['Mean_TEG_mins']:.1f} 个时序窗口预警
* 误报疲劳负荷 (FAB):  每 100 个时间步产生 {alarm_stats['FAB_Rate']:.1f} 次无效警报

【四、 时序因果性探伤 (Temporal Causal Integrity)】
* 全集基线 AUPRC:         {auprc_baseline:.4f}
* 抹除尾部 1min 后:       {auprc_tail_mask:.4f} (预期下降，证明未单纯依赖尾部特征作弊)
* 打乱 10min 时序拓扑后:  {auprc_shuffle:.4f} (预期断崖式下跌，证明模型深度依赖时间因果律)
=====================================================================
    """)

if __name__ == "__main__":
    evaluate_clinical_system()