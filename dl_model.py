"""
Script: dl_model.py
Version: Causal TCN + Energy Envelope + 3-class Rhythm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowEncoder(nn.Module):
    """单窗口形态学编码器 — 直连单通道 ECG，无伪多导联投影。"""

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


class HierarchicalHazardHeads(nn.Module):
    """三层预警头 — 节律 (3cls) / 危急度 (4cls) / 波形稳定性偏离指数 (3 scale)。"""

    def __init__(self, hidden_dim=256):
        super().__init__()
        self.rhythm_head = nn.Linear(hidden_dim, 3)  # Normal, PVC, 室上性心律失常
        self.criticality_head = nn.Linear(hidden_dim, 4)
        self.hazard_deltas = nn.Linear(hidden_dim, 3)
        self.hazard_temperature = nn.Parameter(torch.ones(1))

    def forward_instant(self, h):
        """即时分类 — 当前时刻的节律和危急度。h: [B, hidden_dim]"""
        return (
            self.rhythm_head(h).unsqueeze(1),
            self.criticality_head(h).unsqueeze(1),
        )

    def forward_hazard(self, h):
        """波形稳定性偏离指数 — 三个独立时间尺度，非级联。h: [B, hidden_dim]"""
        raw = self.hazard_deltas(h)
        t = self.hazard_temperature.clamp(min=0.5, max=5.0)
        p30s = torch.sigmoid(raw[..., 0] / t)
        p1m  = torch.sigmoid(raw[..., 1] / t)
        p5m  = torch.sigmoid(raw[..., 2] / t)
        return torch.stack([p30s, p1m, p5m], dim=-1).unsqueeze(1)


class HybridWarningNet(nn.Module):
    """旧版多分类预警模型（遗留兼容）。"""

    def __init__(self, in_channels=1, embed_dim=256, num_classes=6):
        super().__init__()
        self.encoder = WindowEncoder(in_channels, embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return {"logits": self.classifier(self.encoder(x))}


class HierarchicalHazardNet(nn.Module):
    """
    因果 TCN + 能量包络 + RR 分支 (三分类节律)。
    SVT/AT 合并至室上性心律失常。因果卷积+能量包络+双池化+注意力机制。
    """

    def __init__(self, in_channels=1, embed_dim=256, hidden_dim=256, rr_dim=9, rr_hidden=64):
        super().__init__()
        self.window_encoder = WindowEncoder(in_channels, embed_dim)

        # Causal temporal conv: 左填充 only, 不泄露未来信息
        self.tconv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=0)
        self.tbn1 = nn.BatchNorm1d(embed_dim)
        self.tconv2 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=0, dilation=2)
        self.tbn2 = nn.BatchNorm1d(hidden_dim)

        # 双注意力：危急度与预警各自独立查询
        self.crit_attention_query = nn.Linear(hidden_dim, 1, bias=False)
        self.haz_attention_query = nn.Linear(hidden_dim, 1, bias=False)

        self.dropout = nn.Dropout(0.25)  # 平衡 VT 连续性 vs SVT/AT 过拟合

        # RR 特征编码器: rr_dim(9) → rr_hidden(64) → hidden_dim(256)
        self.rr_encoder = nn.Sequential(
            nn.Linear(rr_dim, rr_hidden), nn.ReLU(),
            nn.Linear(rr_hidden, hidden_dim),
        )

        # 能量包络节律分支 — 模拟希尔伯特包络检波, 不依赖 R 峰检测
        # AvgPool1d(k=25,s=25) 将 7500 样本压为 300 个能量脉冲
        self.env_pool = nn.AvgPool1d(kernel_size=25, stride=25)
        self.env_encoder = nn.Sequential(
            nn.Linear(300, 64), nn.ReLU(),
            nn.Linear(64, hidden_dim),
        )

        self.heads = HierarchicalHazardHeads(hidden_dim)

    def forward(self, x_seq, x_rr=None):
        B, S, C, L = x_seq.shape
        flat = x_seq.reshape(B * S, C, L)

        # 分块窗口编码
        CHUNK = 64
        z_chunks = []
        for i in range(0, B * S, CHUNK):
            z_chunks.append(self.window_encoder(flat[i : i + CHUNK]))
            torch.cuda.empty_cache()
        z_all = torch.cat(z_chunks, dim=0)             # [B*S, embed_dim]

        # [B*S, embed_dim] → [B, embed_dim, S] → temporal conv
        z_seq = z_all.view(B, S, -1).transpose(1, 2)   # [B, embed_dim, S]
        h_seq = F.pad(z_seq, (2, 0))                      # k=3,d=1 → left pad 2
        h_seq = F.relu(self.tbn1(self.tconv1(h_seq)))
        h_seq = F.pad(h_seq, (4, 0))                      # k=3,d=2 → left pad 4
        h_seq = F.relu(self.tbn2(self.tconv2(h_seq)))      # [B, hidden_dim, S]
        h_seq = h_seq.transpose(1, 2)                    # [B, S, hidden_dim]

        h_current = h_seq[:, -1, :]                      # [B, hidden_dim]

        # ---- RR 特征支路: 注入节律信息以区分 AFib vs SVT/AT ----
        if x_rr is not None:
            rr_feat = self.rr_encoder(x_rr[:, -1, :])   # [B, hidden_dim]
            h_current = h_current + rr_feat

        # ---- 能量包络支路: abs→包络检波→节律模式 ----
        env = self.env_pool(torch.abs(flat))            # [B*S, 1, 300]
        env = env.view(B * S, -1)                        # [B*S, 300]
        env_feat = self.env_encoder(env)                 # [B*S, hidden_dim]
        env_feat = env_feat.view(B, S, -1)[:, -1, :]    # [B, hidden_dim]
        h_current = h_current + env_feat

        # 危急度注意力：sigmoid 允许多时间点同时激活（归一化防膨胀）
        crit_attn = torch.sigmoid(self.crit_attention_query(h_seq))
        h_crit_ctx = torch.sum(h_seq * crit_attn, dim=1) / (crit_attn.sum(dim=1) + 1e-4)
        h_crit = h_current + h_crit_ctx

        # 预警注意力：独立查询，聚焦前兆窗口
        haz_attn = torch.sigmoid(self.haz_attention_query(h_seq))
        h_hazard = torch.sum(h_seq * haz_attn, dim=1) / (haz_attn.sum(dim=1) + 1e-4)

        # Dropout 防过拟合
        h_current = self.dropout(h_current)
        h_crit = self.dropout(h_crit)
        h_hazard = self.dropout(h_hazard)

        rhythm_logits = self.heads.rhythm_head(h_current).unsqueeze(1)
        criticality_logits = self.heads.criticality_head(h_crit).unsqueeze(1)
        hazard_probs = self.heads.forward_hazard(h_hazard)

        return {
            "preds": {
                "rhythm_logits": rhythm_logits,
                "criticality_logits": criticality_logits,
                "hazard_probs": hazard_probs,
            },
            "h_seq": h_current,
        }
