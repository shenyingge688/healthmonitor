"""
核心机制: Micro-Meso-Macro 架构 | 视界交叉注意力 | 异构结构化推演头
"""
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class WindowEncoder(nn.Module):
    """形态学编码器：提取单窗口特征 (Z_t)，允许高频突变"""
    def __init__(self, in_channels=1):
        super().__init__()
        self.micro_cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, padding=7, stride=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU()
        )
        layers = []
        in_ch = 128
        for i in range(3):
            dilation_size = 2 ** i
            layers.append(TemporalBlock(in_ch, 128, 3, stride=1, dilation=dilation_size, padding=2 * dilation_size))
            in_ch = 128
        self.meso_tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, window_x):
        return self.pool(self.meso_tcn(self.micro_cnn(window_x))).squeeze(-1)

class TemporalCrossAttentionDecoder(nn.Module):
    """时序交叉注意力：保留因果拓扑，按视界 Query 动态查询"""
    def __init__(self, hidden_dim=128):
        super().__init__()
        # 视界 Token [0:30s, 1:1m, 2:5m]
        self.horizon_tokens = nn.Embedding(num_embeddings=3, embedding_dim=hidden_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True, dropout=0.2)

    def forward(self, h_seq, horizon_idx):
        Q = self.horizon_tokens(horizon_idx).unsqueeze(1) # [Batch, 1, 128]
        context_feat, attn_weights = self.cross_attn(query=Q, key=h_seq, value=h_seq)
        return context_feat.squeeze(1), attn_weights.squeeze(1)

class StructuredForecastingHeads(nn.Module):
    """异构多任务预测头：匹配医疗事件真实的随机过程"""
    def __init__(self, hidden_dim=128):
        super().__init__()
        # Head 1: PVC 点过程 (Count Regression -> Poisson Loss)
        self.beat_head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))
        # Head 2: AFib 状态占据 (Occupancy -> BCEWithLogits)
        self.rhythm_head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))
        # Head 3: VT 灾难风险 (Hazard -> Weighted BCE)
        self.hazard_head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))

    def forward(self, context_feat):
        return {
            "pvc_log_rate": self.beat_head(context_feat).squeeze(-1),
            "afib_logits": self.rhythm_head(context_feat).squeeze(-1),
            "vt_hazard_logits": self.hazard_head(context_feat).squeeze(-1)
        }

class LatentDynamicsForecastingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.window_encoder = WindowEncoder(in_channels=1)
        self.macro_gru = nn.GRU(input_size=128, hidden_size=64, num_layers=2, 
                                batch_first=True, bidirectional=True, dropout=0.3)
        self.cross_attn_decoder = TemporalCrossAttentionDecoder(hidden_dim=128)
        self.structured_heads = StructuredForecastingHeads(hidden_dim=128)

    def forward(self, x_seq, horizon_idx):
        batch_size, seq_len, channels, points = x_seq.size()
        z_t_flat = self.window_encoder(x_seq.view(batch_size * seq_len, channels, points))
        z_seq = z_t_flat.view(batch_size, seq_len, 128)
        
        # 提取生理潜状态 h_t
        h_seq, _ = self.macro_gru(z_seq) 
        
        # 视界条件查询与解码
        context_feat, attn_weights = self.cross_attn_decoder(h_seq, horizon_idx)
        preds = self.structured_heads(context_feat)
        
        return {
            "preds": preds,               # 包含 pvc_log_rate, afib_logits, vt_hazard_logits
            "attention": attn_weights,    # [Batch, Seq_Len]
            "h_seq": h_seq                # 暴露给损失函数做平滑正则化
        }