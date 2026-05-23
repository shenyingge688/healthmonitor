"""
Script: dl_model.py
Version: V13.0 — Dual Attention + Rhythm Skip Connection
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
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.stage4(self.stage3(self.stage2(self.stage1(x)))).squeeze(-1)


class HierarchicalHazardHeads(nn.Module):
    """三层预警头 — 节律 (4cls) / 危急度 (4cls) / 崩溃概率 (3 horizon)。"""

    def __init__(self, hidden_dim=256):
        super().__init__()
        self.rhythm_head = nn.Linear(hidden_dim, 4)
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
        """提前预警 — 未来时间窗口的累积崩溃概率。h: [B, hidden_dim]"""
        raw = self.hazard_deltas(h)
        t = self.hazard_temperature.clamp(min=0.5, max=5.0)
        p30s = torch.sigmoid(raw[..., 0] / t)
        p1m = p30s + (1.0 - p30s) * torch.sigmoid(raw[..., 1] / t)
        p5m = p1m + (1.0 - p1m) * torch.sigmoid(raw[..., 2] / t)
        return torch.stack([p30s, p1m, p5m], dim=-1).unsqueeze(1)


class HybridWarningNet(nn.Module):
    """旧版 V5 多分类预警模型（遗留兼容）。"""

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
    V13 — Dual Attention (危急度/预警分流) + Rhythm 直连跳过 temporal_conv。
    根治标签泄漏和节律准确率下降。
    """

    def __init__(self, in_channels=1, embed_dim=256, hidden_dim=256):
        super().__init__()
        self.window_encoder = WindowEncoder(in_channels, embed_dim)

        # 时序卷积提炼趋势（无池化，保留 19 个时间步）
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim), nn.ReLU(),
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(),
        )

        # 双注意力：危急度与预警各自独立查询
        self.crit_attention_query = nn.Linear(hidden_dim, 1, bias=False)
        self.haz_attention_query = nn.Linear(hidden_dim, 1, bias=False)

        self.dropout = nn.Dropout(0.1)

        self.heads = HierarchicalHazardHeads(hidden_dim)

    def forward(self, x_seq):
        B, S, C, L = x_seq.shape
        flat = x_seq.reshape(B * S, C, L)

        # 分块窗口编码
        CHUNK = 32  # 190 windows → 6 chunks
        z_chunks = []
        for i in range(0, B * S, CHUNK):
            z_chunks.append(self.window_encoder(flat[i : i + CHUNK]))
            torch.cuda.empty_cache()
        z_all = torch.cat(z_chunks, dim=0)             # [B*S, 256]

        # [B*S, 256] → [B, 256, S] → temporal conv
        z_seq = z_all.view(B, S, -1).transpose(1, 2)   # [B, 256, S]
        h_seq = self.temporal_conv(z_seq)                # [B, hidden_dim, S]
        h_seq = h_seq.transpose(1, 2)                    # [B, S, hidden_dim]

        h_current = h_seq[:, -1, :]                      # [B, hidden_dim]  — 当前时刻

        # 危急度注意力：独立查询，不混入 hazard 信号
        crit_attn = F.softmax(self.crit_attention_query(h_seq), dim=1)
        h_crit_ctx = torch.sum(h_seq * crit_attn, dim=1)
        h_crit = h_current + h_crit_ctx                  # [B, hidden_dim]

        # 预警注意力：独立查询，聚焦前兆窗口
        haz_attn = F.softmax(self.haz_attention_query(h_seq), dim=1)
        h_hazard = torch.sum(h_seq * haz_attn, dim=1)    # [B, hidden_dim]

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
