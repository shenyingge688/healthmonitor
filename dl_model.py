"""
Script: dl_model.py
Version: V11.0 (TemporalConv + Pool replaces GRU — single-stage compatible)
"""
import torch
import torch.nn as nn


class WindowEncoder(nn.Module):
    """提取单窗口的形态学特征：1-channel ECG → 256-dim embedding。"""

    def __init__(self, in_channels=12, embed_dim=256):
        super().__init__()
        self.lead_projector = nn.Conv1d(1, 12, kernel_size=1, bias=False)

        self.stage1 = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7),
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
        if x.size(1) == 1:
            x = self.lead_projector(x)
        return self.stage4(self.stage3(self.stage2(self.stage1(x)))).squeeze(-1)


class HierarchicalHazardHeads(nn.Module):
    """三层临床预警头：节律 → 危急度 → 时间窗口崩溃概率。"""

    def __init__(self, hidden_dim=256):
        super().__init__()
        self.rhythm_head = nn.Linear(hidden_dim, 4)
        self.criticality_head = nn.Linear(hidden_dim, 4)
        self.hazard_deltas = nn.Linear(hidden_dim, 3)
        self.hazard_temperature = nn.Parameter(torch.ones(1))

    def forward(self, h):
        # h: [B, hidden_dim]  或  [B, 1, hidden_dim]
        if h.dim() == 3:
            h = h.squeeze(1)

        rhythm_logits = self.rhythm_head(h)
        criticality_logits = self.criticality_head(h)

        raw_deltas = self.hazard_deltas(h)
        t = self.hazard_temperature.clamp(min=0.2, max=5.0)
        p30s = torch.sigmoid(raw_deltas[..., 0] / t)
        p1m = p30s + (1.0 - p30s) * torch.sigmoid(raw_deltas[..., 1] / t)
        p5m = p1m + (1.0 - p1m) * torch.sigmoid(raw_deltas[..., 2] / t)
        hazard_probs = torch.stack([p30s, p1m, p5m], dim=-1)

        # 保持与旧版兼容的维度：[B, 1, num_classes] 和 [B, 1, 3]
        return {
            "rhythm_logits": rhythm_logits.unsqueeze(1),
            "criticality_logits": criticality_logits.unsqueeze(1),
            "hazard_probs": hazard_probs.unsqueeze(1),
        }


class HybridWarningNet(nn.Module):
    """旧版 V5 多分类预警模型（遗留兼容）。"""

    def __init__(self, in_channels=1, embed_dim=256, num_classes=6):
        super().__init__()
        self.encoder = WindowEncoder(in_channels, embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        feat = self.encoder(x)
        return {"logits": self.classifier(feat)}


class HierarchicalHazardNet(nn.Module):
    """V11 — TemporalConv + AdaptivePool 替代 GRU，单阶段可训练。"""

    def __init__(self, in_channels=12, embed_dim=256, hidden_dim=256):
        super().__init__()
        self.window_encoder = WindowEncoder(in_channels, embed_dim)

        # 轻量时序卷积替代 GRU — 双层膨胀卷积
        # Layer 1: kernel_size=5, dilation=1 → ±30s 局部上下文
        # Layer 2: kernel_size=5, dilation=5 → ±375s 全局上下文（覆盖 5 分钟预警窗口）
        self.temporal_pool = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2, dilation=1),
            nn.BatchNorm1d(embed_dim), nn.ReLU(),
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=5, padding=10, dilation=5),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.heads = HierarchicalHazardHeads(hidden_dim)

    def forward(self, x_seq):
        B, S, C, L = x_seq.shape
        flat = x_seq.reshape(B * S, C, L)

        # 分块编码：防止 8GB 显存 OOM
        CHUNK = 16  # 无 GRU 后编码器轻量，可加大块
        z_chunks = []
        for i in range(0, B * S, CHUNK):
            z_chunks.append(self.window_encoder(flat[i : i + CHUNK]))
            torch.cuda.empty_cache()
        z_all = torch.cat(z_chunks, dim=0)          # [B*S, 256]

        # [B*S, 256] → [B, S, 256] → [B, 256, S] → Conv1d over time axis
        z_seq = z_all.view(B, S, -1).transpose(1, 2)  # [B, 256, S]
        h_pooled = self.temporal_pool(z_seq).squeeze(-1)  # [B, hidden_dim]

        return {"preds": self.heads(h_pooled), "h_seq": h_pooled}
