"""
Script: dl_model.py
Version: V10.0 (Hierarchical Hazard Network)
"""
import torch
import torch.nn as nn

class WindowEncoder(nn.Module):
    def __init__(self, in_channels=12, embed_dim=256):
        super().__init__()
        self.lead_projector = nn.Conv1d(1, 12, kernel_size=1, bias=False) 
        
        self.stage1 = nn.Sequential(nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        self.stage2 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        self.stage3 = nn.Sequential(nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2), nn.BatchNorm1d(256), nn.ReLU())
        self.stage4 = nn.Sequential(nn.Conv1d(256, embed_dim, kernel_size=3, stride=2, padding=1), nn.BatchNorm1d(embed_dim), nn.ReLU(), nn.AdaptiveAvgPool1d(1))

    def forward(self, x):
        if x.size(1) == 1: x = self.lead_projector(x) 
        return self.stage4(self.stage3(self.stage2(self.stage1(x)))).squeeze(-1)

class HierarchicalHazardHeads(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        # Level 1: Rhythm Organization (0:Normal, 1:PVC, 2:AFIB, 3:SVT/AT)
        self.rhythm_head = nn.Linear(hidden_dim, 4)
        
        # Level 2: Ventricular Criticality (0:Stable, 1:Instability, 2:VT, 3:VF)
        self.criticality_head = nn.Linear(hidden_dim, 4)
        
        # Level 3: Discrete Cumulative Hazard (30s, 1m, 5m 增量)
        self.hazard_deltas = nn.Linear(hidden_dim, 3) 

    def forward(self, h_seq):
        rhythm_logits = self.rhythm_head(h_seq)
        criticality_logits = self.criticality_head(h_seq)
        
        # 核心：累积风险单调性约束 P(30s) <= P(1m) <= P(5m)
        raw_deltas = self.hazard_deltas(h_seq)
        
        p30s = torch.sigmoid(raw_deltas[..., 0])
        p1m = p30s + (1.0 - p30s) * torch.sigmoid(raw_deltas[..., 1])
        p5m = p1m + (1.0 - p1m) * torch.sigmoid(raw_deltas[..., 2])
        
        hazard_probs = torch.stack([p30s, p1m, p5m], dim=-1)

        return {
            "rhythm_logits": rhythm_logits,
            "criticality_logits": criticality_logits,
            "hazard_probs": hazard_probs
        }

class HierarchicalHazardNet(nn.Module):
    def __init__(self, in_channels=12, embed_dim=256, hidden_dim=256):
        super().__init__()
        self.window_encoder = WindowEncoder(in_channels, embed_dim)
        self.macro_gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=False)
        self.heads = HierarchicalHazardHeads(hidden_dim)

    def forward(self, x_seq):
        B, S, C, L = x_seq.shape
        z_seq = self.window_encoder(x_seq.view(B * S, C, L)).view(B, S, -1)
        h_seq, _ = self.macro_gru(z_seq)
        return {"preds": self.heads(h_seq), "h_seq": h_seq}