"""
Script: dl_model.py
Version: V8.3 Final
"""
import torch
import torch.nn as nn

class WindowEncoder(nn.Module):
    def __init__(self, in_channels=12, embed_dim=256):
        super(WindowEncoder, self).__init__()
        # 空间护城河：单导联投射
        self.lead_projector = nn.Conv1d(1, 12, kernel_size=1, bias=False) 
        nn.init.xavier_uniform_(self.lead_projector.weight)
        
        self.stage1 = nn.Sequential(nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        self.stage2 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        self.stage3 = nn.Sequential(nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2), nn.BatchNorm1d(256), nn.ReLU())
        self.stage4 = nn.Sequential(nn.Conv1d(256, embed_dim, kernel_size=3, stride=2, padding=1), nn.BatchNorm1d(embed_dim), nn.ReLU(), nn.AdaptiveAvgPool1d(1))

    def forward(self, x):
        if x.size(1) == 1:
            x = self.lead_projector(x) 
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x.squeeze(-1)

class StructuredHeads(nn.Module):
    def __init__(self, hidden_dim=256):
        super(StructuredHeads, self).__init__()
        self.pvc_head = nn.Linear(hidden_dim, 1)
        self.afib_head = nn.Linear(hidden_dim, 1)
        self.vt_hazard_head = nn.Linear(hidden_dim, 3) 

    def forward(self, h_seq):
        return {
            "pvc_logits": self.pvc_head(h_seq).squeeze(-1),
            "afib_logits": self.afib_head(h_seq).squeeze(-1),
            "vt_hazard_logits": self.vt_hazard_head(h_seq) 
        }

class LatentDynamicsForecastingNet(nn.Module):
    def __init__(self, in_channels=12, seq_len=19, embed_dim=256, hidden_dim=256):
        super(LatentDynamicsForecastingNet, self).__init__()
        self.window_encoder = WindowEncoder(in_channels, embed_dim)
        
        # 因果护城河：单向强制
        self.macro_gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=False)
        self.structured_heads = StructuredHeads(hidden_dim)

    def forward(self, x_seq):
        B, S, C, L = x_seq.shape
        x_flat = x_seq.view(B * S, C, L)
        z_flat = self.window_encoder(x_flat)
        z_seq = z_flat.view(B, S, -1)
        
        h_seq, _ = self.macro_gru(z_seq)
        preds = self.structured_heads(h_seq)
        return {"preds": preds, "h_seq": h_seq}