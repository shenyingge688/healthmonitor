"""
Script: dl_model.py
心电监护预警网络 — 超前 5 分钟心律失常预测

架构融合：
  1. WindowEncoder — 4 阶段 CNN 骨干预训练 (PTB-XL)，单窗口形态学编码
  2. Global Skip TCN — 3 层因果时序卷积 + 多尺度跳跃连接汇总
  3. RR 间期特征分支 — 样本熵 + 庞加莱图，区分房颤与规则性室上速
  4. 能量包络分支 — 模拟希尔伯特包络检波，节律模式感知
  5. Sigmoid 时序注意力 — 产生 1D-CAM 可解释热力图
  6. 单一 6 分类输出 — 未来 5 分钟心律失常预测
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 基础组件
# ==========================================

class Chomp1d(nn.Module):
    """因果裁剪：移除右侧填充，确保时序不泄露未来信息"""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """因果时序卷积块 — 同时返回主路径和跳跃连接输出"""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        main_out = self.relu(out + res)
        skip_out = out  # 纯净特征，未经残差叠加
        return main_out, skip_out


# ==========================================
# 窗口编码器 (PTB-XL 预训练骨干)
# ==========================================

class WindowEncoder(nn.Module):
    """单窗口形态学编码器 — 4 阶段 CNN，输出 256 维嵌入"""

    def __init__(self, in_channels=1, embed_dim=256):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.stage2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.stage3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.stage4 = nn.Sequential(
            nn.Conv1d(256, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(embed_dim), nn.ReLU(),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.pool_proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x):
        x = self.stage4(self.stage3(self.stage2(self.stage1(x))))
        avg = self.avg_pool(x).squeeze(-1)
        max_f = self.max_pool(x).squeeze(-1)
        return self.pool_proj(torch.cat([avg, max_f], dim=-1))


# ==========================================
# 核心预警网络
# ==========================================

class ArrhythmiaWarningNet(nn.Module):
    """
    超前 5 分钟心律失常预测网络。

    输入: ECG 轨迹 [B, N_WINDOWS, 1, PTS_PER_WIN]
          RR 特征 [B, N_WINDOWS, 9]（可选）
    输出: 6 分类 logits + 1D-CAM 热力图
    """

    def __init__(self,
                 in_channels=1,
                 embed_dim=256,
                 hidden_dim=256,
                 n_windows=39,
                 rr_dim=9,
                 rr_hidden=64,
                 num_classes=6,
                 tcn_channels=(256, 256, 256),
                 dropout=0.2):
        super().__init__()

        # ---- 1. 窗口编码器 (PTB-XL 预训练) ----
        self.window_encoder = WindowEncoder(in_channels, embed_dim)

        # ---- 2. Global Skip TCN ----
        tcn_layers = []
        skip_adapters = []
        in_ch = embed_dim
        for i, out_ch in enumerate(tcn_channels):
            dilation = 2 ** i
            tcn_layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size=3, stride=1,
                              dilation=dilation, dropout=dropout))
            if i < len(tcn_channels) - 1:
                skip_adapters.append(nn.Conv1d(out_ch, hidden_dim, 1))
            in_ch = out_ch
        self.tcn_blocks = nn.ModuleList(tcn_layers)
        self.skip_adapters = nn.ModuleList(skip_adapters)

        # ---- 3. 时序注意力 (1D-CAM) ----
        self.temporal_attn = nn.Sequential(
            nn.Conv1d(hidden_dim, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # ---- 4. RR 间期特征编码 ----
        self.rr_encoder = nn.Sequential(
            nn.Linear(rr_dim, rr_hidden), nn.ReLU(),
            nn.Linear(rr_hidden, hidden_dim),
        )

        # ---- 5. 能量包络分支 ----
        self.env_pool = nn.AvgPool1d(kernel_size=25, stride=25)
        self.env_encoder = nn.Sequential(
            nn.Linear(300, 64), nn.ReLU(),
            nn.Linear(64, hidden_dim),
        )

        # ---- 6. 分类头 (6 类) ----
        self.class_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_seq, x_rr=None):
        """
        Args:
            x_seq: [B, S, C, L]   S=窗口数, C=1, L=窗口采样点
            x_rr:  [B, S, 9]      可选 RR 特征
        Returns:
            dict with logits, prob, cam
        """
        B, S, C, L = x_seq.shape
        flat = x_seq.reshape(B * S, C, L)

        # ---- 步骤 1: 窗口编码 ----
        CHUNK = 64
        z_chunks = []
        for i in range(0, B * S, CHUNK):
            z_chunks.append(self.window_encoder(flat[i : i + CHUNK]))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        z_all = torch.cat(z_chunks, dim=0)  # [B*S, embed_dim]

        # ---- 步骤 2: Global Skip TCN ----
        t_in = z_all.view(B, S, -1).transpose(1, 2)  # [B, embed_dim, S]
        global_skip = 0
        for i, block in enumerate(self.tcn_blocks):
            t_in, skip_out = block(t_in)
            if i < len(self.skip_adapters):
                global_skip = global_skip + self.skip_adapters[i](skip_out)
            else:
                # 最后一层 skip 直接加入，无需适配器（通道已对齐）
                if skip_out.size(1) == self.skip_adapters[-1].out_channels:
                    global_skip = global_skip + skip_out
                else:
                    # fallback: 使用倒数第二个适配器
                    global_skip = global_skip + self.skip_adapters[-1](skip_out)

        # ---- 步骤 3: 注意力加权池化 ----
        attn_weights = self.temporal_attn(global_skip)         # [B, 1, S_tcn]
        weighted_feat = torch.sum(global_skip * attn_weights, dim=2) \
                        / (attn_weights.sum(dim=2) + 1e-4)    # [B, hidden_dim]

        # ---- 步骤 4: RR 特征注入 ----
        if x_rr is not None:
            rr_feat = self.rr_encoder(x_rr[:, -1, :])          # [B, hidden_dim]
            weighted_feat = weighted_feat + rr_feat

        # ---- 步骤 5: 能量包络注入 ----
        env = self.env_pool(torch.abs(flat))                   # [B*S, 1, 300]
        env = env.view(B * S, -1)
        env_feat = self.env_encoder(env)                        # [B*S, hidden_dim]
        env_feat = env_feat.view(B, S, -1).mean(dim=1)         # [B, hidden_dim]
        weighted_feat = weighted_feat + env_feat

        # ---- 步骤 6: 分类 ----
        logits = self.class_head(weighted_feat)                 # [B, 6]

        # ---- 步骤 7: 1D-CAM 热力图 ----
        _hd = weighted_feat.size(-1)
        cam_weights = self.class_head[1].weight[:, :_hd].mean(dim=0)
        cam_1d = torch.relu(torch.sum(
            cam_weights.view(1, _hd, 1) * global_skip, dim=1))
        cam_1d = cam_1d / (cam_1d.max(dim=1, keepdim=True)[0] + 1e-8)

        return {
            "logits": logits,
            "prob": torch.softmax(logits, dim=1),
            "cam": cam_1d,
        }
