"""
Script: dl_model.py
Version: V8.0 (Discrete-Time Survival Architecture)
Description: 
包含 PTFN 的核心动力学推演架构。
在 V8.0 中，vt_hazard_head 被升级为输出 3 个离散区间的条件风险率 (Conditional Hazards)。
底层 CNN 依然严格划分为 stage1 ~ stage4。
"""

import torch
import torch.nn as nn

class WindowEncoder(nn.Module):
    """
    1D-CNN 编码器：将 10 秒心电图切片压缩为高维形态学表征。
    """
    def __init__(self, in_channels=12, embed_dim=256):
        super(WindowEncoder, self).__init__()
        
        # 永久冻结区：基础边缘与波形图元
        self.stage1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.stage2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        # 域适应微调区：高层病理语义
        self.stage3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.stage4 = nn.Sequential(
            nn.Conv1d(256, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x.squeeze(-1)


class CrossAttnDecoder(nn.Module):
    """视界因果查询解码器"""
    def __init__(self, hidden_dim=256):
        super(CrossAttnDecoder, self).__init__()
        self.horizon_embedding = nn.Embedding(5, hidden_dim) 
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h_seq, horizon_idx):
        Q = self.horizon_embedding(horizon_idx).unsqueeze(1)
        context_vector, _ = self.attention(query=Q, key=h_seq, value=h_seq)
        context_vector = self.layer_norm(context_vector + Q)
        return context_vector.squeeze(1)


class StructuredHeads(nn.Module):
    """
    多任务临床终点输出。
    V8.0 核心升级：VT 预测头从单一标量升级为 3-Bin Hazard 序列。
    """
    def __init__(self, hidden_dim=256):
        super(StructuredHeads, self).__init__()
        self.pvc_head = nn.Linear(hidden_dim, 1)
        self.afib_head = nn.Linear(hidden_dim, 1)
        
        # 👇【V8.0 核心重构】：输出 3 个维度的 Hazard (h1_30s, h2_2m, h3_5m)
        self.vt_hazard_head = nn.Linear(hidden_dim, 3) 

    def forward(self, c_vector):
        preds = {
            "pvc_log_rate": self.pvc_head(c_vector).squeeze(-1),
            "afib_logits": self.afib_head(c_vector).squeeze(-1),
            # 注意：这里去掉了 squeeze(-1)，因为现在是 [Batch, 3] 形状
            "vt_hazard_logits": self.vt_hazard_head(c_vector) 
        }
        return preds


class LatentDynamicsForecastingNet(nn.Module):
    def __init__(self, in_channels=12, seq_len=10, embed_dim=256, hidden_dim=256):
        super(LatentDynamicsForecastingNet, self).__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.window_encoder = WindowEncoder(in_channels, embed_dim)
        self.macro_gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.cross_attn_decoder = CrossAttnDecoder(hidden_dim)
        self.structured_heads = StructuredHeads(hidden_dim)

    def forward(self, x_seq, horizon_idx):
        B, S, C, L = x_seq.shape
        x_flat = x_seq.view(B * S, C, L)
        z_flat = self.window_encoder(x_flat)
        z_seq = z_flat.view(B, S, self.embed_dim)
        h_seq, _ = self.macro_gru(z_seq)
        c_vector = self.cross_attn_decoder(h_seq, horizon_idx)
        preds = self.structured_heads(c_vector)
        
        return {
            "preds": preds,
            "h_seq": h_seq
        }