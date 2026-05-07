"""
HealthMonitor V4.0 - 深度学习网络模型主体架构 (Micro-Meso-Macro 骨架)
"""
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
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
        return self.relu(out + res), out

# ==========================================
# [V5.0 架构预留] 宏观长程特征编码器
# ==========================================
class MacroTransformerEncoder(nn.Module):
    """
    预留组件：处理 1~24 小时 HRV 序列、基线漂移趋势。
    V4.0 采用轻量 MLP 占位避免 OOM，V5.0 替换为标准 TransformerEncoder。
    """
    def __init__(self, feature_dim=32):
        super().__init__()
        self.placeholder_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )

    def forward(self, macro_features):
        return self.placeholder_net(macro_features)

# ==========================================
# 主干网络: HybridWarningNet
# ==========================================
class HybridWarningNet(nn.Module):
    def __init__(self, in_channels=1, hrv_feature_dim=10):
        super().__init__()
        # 1. 微观层 (Micro): CNN 提取波形形态
        self.cnn_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=15, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 2. 中观层 (Meso): TCN 提取节律演变
        num_channels = [32, 64, 64]
        layers = []
        in_ch = 32
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            layers.append(TemporalBlock(in_ch, num_channels[i], 3, stride=1, dilation=dilation_size, padding=2 * dilation_size))
            in_ch = num_channels[i]
        self.tcn_blocks = nn.ModuleList(layers)
        self.skip_adapters = nn.ModuleList([nn.Conv1d(32, 64, 1), nn.Conv1d(64, 64, 1)])
        self.temporal_attn = nn.Sequential(nn.Conv1d(64, 16, 1), nn.ReLU(), nn.Conv1d(16, 1, 1), nn.Sigmoid())
        
        self.rhythm_pool = nn.AvgPool1d(kernel_size=25, stride=25)
        self.rhythm_linear = nn.Linear(20, 8)

        # 3. 宏观层 (Macro): Transformer 旁路预留
        self.hrv_proj = nn.Linear(hrv_feature_dim, 32)
        self.macro_transformer = MacroTransformerEncoder(feature_dim=32)
        
        # 4. 多任务双输出头 (Multi-Task Learning)
        # 融合维度: TCN(64) + Rhythm(8) + Transformer(32) = 104
        self.future_warning_head = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(104, 32), nn.ReLU(), nn.Linear(32, 6)
        )
        # 辅助任务头：诊断当前 10 分钟的病情，作为超前预警的跳板
        self.current_state_head = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(104, 32), nn.ReLU(), nn.Linear(32, 6)
        )

    def forward(self, x, hrv_features=None):
        batch_size, seq_len, channels, points = x.size()
        c_in = x.view(batch_size * seq_len, channels, points)
        
        # Micro
        cnn_features = self.cnn_extractor(c_in)
        c_out = torch.cat([self.max_pool(cnn_features).squeeze(-1), self.avg_pool(cnn_features).squeeze(-1)], dim=1)
        
        # Meso
        t_in = c_out.view(batch_size, seq_len, 32).transpose(1, 2)
        global_skip = 0
        for i, block in enumerate(self.tcn_blocks):
            t_in, skip_out = block(t_in)
            global_skip += self.skip_adapters[i](skip_out) if i < len(self.skip_adapters) else skip_out
            
        attn_weights = self.temporal_attn(global_skip)
        weighted_feat = torch.sum(global_skip * attn_weights, dim=2) / (attn_weights.sum(dim=2) + 1e-4)
        
        envelope = self.rhythm_pool(torch.abs(c_in)).view(batch_size * seq_len, -1)
        rhythm_feat = self.rhythm_linear(envelope).view(batch_size, seq_len, 8).mean(dim=1)
        
        # Macro
        if hrv_features is not None:
            macro_emb = self.hrv_proj(hrv_features)
            macro_feat = self.macro_transformer(macro_emb)
        else:
            macro_feat = torch.zeros(batch_size, 32, device=x.device)
            
        combined_feat = torch.cat([weighted_feat, rhythm_feat, macro_feat], dim=1)
        
        # 双重预测输出
        logits_future = self.future_warning_head(combined_feat)
        logits_current = self.current_state_head(combined_feat)
        
        cam_weights = self.future_warning_head[1].weight[:, :64].mean(dim=0)
        cam_1d = torch.relu(torch.sum(cam_weights.view(1, 64, 1) * global_skip, dim=1))
        
        return {
            "logits": logits_future,                   # 超前预警输出 (主任务)
            "logits_current": logits_current,          # 当前状态诊断 (辅任务)
            "prob": torch.softmax(logits_future, dim=1),
            "cam": cam_1d / (cam_1d.max(dim=1, keepdim=True)[0] + 1e-8)
        }