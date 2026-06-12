"""
Script: dl_model.py
ECG early-warning network V6 - Dual-head architecture

Architecture:
  1. WindowEncoder - 4-stage CNN (PTB-XL pretrained optional, frozen by default)
  2. Global Skip TCN - 3-layer causal dilated conv
  3. RR feature branch - 39-window RR statistics
  4. Energy envelope branch - rhythm perception

Dual heads:
  - Current state head: classifies current rhythm (6 classes, CrossEntropy)
  - Future tendency head: predicts future 2-5min state distribution (6-dim, KLDiv)

Input:  ECG [B, 39, 1, 7500] + RR [B, 39, 9]
Output: logits_cur, probs_cur, probs_fut, cam
"""
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
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
        return self.relu(out + res), out


class WindowEncoder(nn.Module):
    def __init__(self, in_channels=1, embed_dim=256):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        self.stage2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        self.stage3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256), nn.ReLU())
        self.stage4 = nn.Sequential(
            nn.Conv1d(256, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(embed_dim), nn.ReLU())
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.pool_proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x):
        x = self.stage4(self.stage3(self.stage2(self.stage1(x))))
        avg = self.avg_pool(x).squeeze(-1)
        max_f = self.max_pool(x).squeeze(-1)
        return self.pool_proj(torch.cat([avg, max_f], dim=-1))


class ArrhythmiaWarningNet(nn.Module):
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
        self.hidden_dim = hidden_dim
        self.encoder_grad_enabled = False

        self.window_encoder = WindowEncoder(in_channels, embed_dim)

        tcn_layers, skip_adapters = [], []
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

        self.temporal_attn = nn.Sequential(
            nn.Conv1d(hidden_dim, 32, kernel_size=1), nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1), nn.Sigmoid())

        self.rr_encoder = nn.Sequential(
            nn.Linear(rr_dim, rr_hidden), nn.ReLU(),
            nn.Linear(rr_hidden, hidden_dim))

        self.env_pool = nn.AvgPool1d(kernel_size=25, stride=25)
        self.env_encoder = nn.Sequential(
            nn.Linear(300, 64), nn.ReLU(),
            nn.Linear(64, hidden_dim))

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout))

        # V6: dual heads
        self.current_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128), nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes))

        self.future_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128), nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes))

    def freeze_encoder(self):
        for p in self.window_encoder.parameters():
            p.requires_grad = False
        self.encoder_grad_enabled = False
        # Keep BatchNorm running stats fixed at pretrained values: running under
        # no_grad does NOT stop BN buffer updates, so eval() is required.
        self.window_encoder.eval()

    def unfreeze_encoder_tail(self):
        """Fine-tune only the final CNN stage and projection of the PTB-XL encoder."""
        for p in self.window_encoder.parameters():
            p.requires_grad = False
        for module in (self.window_encoder.stage4, self.window_encoder.pool_proj):
            for p in module.parameters():
                p.requires_grad = True
        self.encoder_grad_enabled = True
        # Keep BatchNorm running stats fixed while allowing affine weights to adapt.
        self.window_encoder.eval()

    def forward(self, x_seq, x_rr=None):
        B, S, C, L = x_seq.shape
        device = x_seq.device

        encoder_ctx = nullcontext() if (self.encoder_grad_enabled and torch.is_grad_enabled()) else torch.no_grad()
        with encoder_ctx:
            flat = x_seq.reshape(B * S, C, L)
            CHUNK = 64
            z_chunks = []
            for i in range(0, B * S, CHUNK):
                z_chunks.append(self.window_encoder(flat[i:i+CHUNK]))
            z_all = torch.cat(z_chunks, dim=0)

        t_in = z_all.view(B, S, -1).transpose(1, 2)
        global_skip = 0
        for i, block in enumerate(self.tcn_blocks):
            t_in, skip_out = block(t_in)
            if i < len(self.skip_adapters):
                global_skip = global_skip + self.skip_adapters[i](skip_out)
            else:
                if skip_out.size(1) == self.skip_adapters[-1].out_channels:
                    global_skip = global_skip + skip_out
                else:
                    global_skip = global_skip + self.skip_adapters[-1](skip_out)

        attn = self.temporal_attn(global_skip)
        time_feat = (global_skip * attn).sum(dim=2) / (attn.sum(dim=2) + 1e-4)

        if x_rr is not None:
            rr_feat = self.rr_encoder(x_rr[:, -1, :])
        else:
            rr_feat = torch.zeros(B, self.hidden_dim, device=device)

        env = self.env_pool(torch.abs(flat))
        env_feat = self.env_encoder(env.view(B * S, -1))
        env_feat = env_feat.view(B, S, -1).mean(dim=1)

        combined = torch.cat([time_feat, rr_feat, env_feat], dim=-1)
        fused = self.fusion(combined)

        logits_cur = self.current_head(fused)
        logits_fut = self.future_head(fused)

        # Serving CAM = the TRAINED temporal attention (supervised via the
        # attention-pooled time_feat that feeds both heads), not a dead random
        # projection. Normalized per-sample to [0,1] over the 39 windows.
        cam_out = attn.squeeze(1)  # [B, S_tcn]
        cam_out = cam_out / (cam_out.max(dim=1, keepdim=True)[0] + 1e-8)

        return {
            "logits_cur": logits_cur,
            "probs_cur": F.softmax(logits_cur, dim=-1),
            "logits_fut": logits_fut,
            "probs_fut": F.softmax(logits_fut, dim=-1),
            "fused": fused,
            "cam": cam_out,
            "global_skip": global_skip,   # exposed for offline Grad-CAM
        }

    def grad_cam(self, x_seq, x_rr, target_class, head="future"):
        """Class-conditional Grad-CAM over the 39 history windows.

        Backprops the target-class logit to global_skip (the TCN feature map),
        weights channels by their mean gradient, ReLU, normalizes. Returns a
        [B, S_tcn] saliency that is specific to target_class — unlike the dead
        averaged cam_layer it replaces.
        """
        self.eval()
        x_seq = x_seq.clone().requires_grad_(False)
        out = self.forward(x_seq, x_rr=x_rr)
        gs = out["global_skip"]                       # [B, hidden, S]
        logits = out["logits_fut"] if head == "future" else out["logits_cur"]
        score = logits[:, target_class].sum()
        grads = torch.autograd.grad(score, gs, retain_graph=False, create_graph=False)[0]
        weights = grads.mean(dim=2, keepdim=True)     # [B, hidden, 1] channel importance
        cam = torch.relu((weights * gs).sum(dim=1))   # [B, S]
        cam = cam / (cam.amax(dim=1, keepdim=True) + 1e-8)
        return cam.detach()
