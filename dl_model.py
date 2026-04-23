"""
模块名称：多模态混合时序预警网络 (Hybrid Warning Network V3.0)
模块功能：
    1. 局部特征提取：利用 1D-CNN 捕获瞬时心电形态畸变。
    2. 全局时序建模：利用 TCN (时序卷积网络) 捕捉 10 分钟长程演变，并支持全局跳跃连接 (Global Skip)。
    3. 稀疏时序注意力：利用 Sigmoid 激活的注意力机制准确定位病灶瞬间。
    4. 临床节律感知：专门设计的能量包络提取分支，增强对节律失常（如房颤、早搏间期）的感知。
    5. 可解释性支持：内置 1D-CAM 计算，支持 ONNX 导出，实时输出病灶热力图。
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

# ==========================================
# 基础时序组件
# ==========================================

class Chomp1d(nn.Module):
    """裁剪模块：移除卷积产生的对称 Padding，确保时序因果性"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """
    改进型时序卷积块：
    同时返回主路径输出 (Main Path) 和 跳跃连接输出 (Skip Connection)。
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        # 计算主路径输出
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        main_out = self.relu(out + res)
        
        # 提取跳跃连接特征 (Skip Output)
        # 物理意义：当前层新提取出的、未经过残差叠加的“纯净”特征模式
        skip_out = out 
        
        return main_out, skip_out

# ==========================================
# 核心预警网络架构
# ==========================================

class HybridWarningNet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        
        # --- 1. 局部形态特征流 (1D-CNN) ---
        self.cnn_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=15, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # --- 2. 时序演变建模流 (TCN) ---
        num_channels = [32, 64, 64]
        kernel_size = 3
        layers = []
        in_ch = 32
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            out_ch = num_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1, 
                                        dilation=dilation_size, padding=(kernel_size-1) * dilation_size))
            in_ch = out_ch
        self.tcn_blocks = nn.ModuleList(layers)

        # 适配器：将不同深度的 TCN 跳跃输出统一至 64 通道
        self.skip_adapters = nn.ModuleList([
            nn.Conv1d(32, 64, 1),
            nn.Conv1d(64, 64, 1)
        ])

        # --- 3. 稀疏时序注意力模块 ---
        self.temporal_attn = nn.Sequential(
            nn.Conv1d(64, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1),
            nn.Sigmoid() 
        )

        # --- 4. 临床节律感知分支 (能量包络提取) ---
        # 捕捉 RR 间期变化，模拟希尔伯特变换的包络检波能力
        self.rhythm_pool = nn.AvgPool1d(kernel_size=25, stride=25) 
        self.rhythm_linear = nn.Linear(20, 8) 

        # --- 5. 最终决策分类头 ---
        # 输入维度 = 64 (主干时序) + 8 (节律分支) = 72
        self.warning_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(72, 16),
            nn.ReLU(),
            nn.Linear(16, 1) 
        )

    def forward(self, x):
        batch_size, seq_len, channels, points = x.size()
        c_in = x.view(batch_size * seq_len, channels, points)

        # --- 步骤 1: 提取局部窗口特征 ---
        cnn_features = self.cnn_extractor(c_in)
        max_feats = self.max_pool(cnn_features).squeeze(-1)
        avg_feats = self.avg_pool(cnn_features).squeeze(-1)
        c_out = torch.cat([max_feats, avg_feats], dim=1) 

        # --- 步骤 2: 全局时序卷积与跳跃连接汇总 ---
        t_in = c_out.view(batch_size, seq_len, 32).transpose(1, 2)
        global_skip = 0
        for i, block in enumerate(self.tcn_blocks):
            # 🟢 关键：现在 TemporalBlock 会返回两个张量
            t_in, skip_out = block(t_in)
            if i < len(self.skip_adapters):
                global_skip += self.skip_adapters[i](skip_out)
            else:
                global_skip += skip_out 

        # --- 步骤 3: 计算注意力权重并执行加权池化 ---
        attn_weights = self.temporal_attn(global_skip)
        weighted_feat = torch.sum(global_skip * attn_weights, dim=2) / (attn_weights.sum(dim=2) + 1e-4)

        # --- 步骤 4: 计算原始信号的能量包络 (节律感知) ---
        # 使用 torch.abs(c_in) 代替非标准 nn.Abs()
        envelope = self.rhythm_pool(torch.abs(c_in)) 
        envelope = envelope.view(batch_size * seq_len, -1)
        rhythm_feat = self.rhythm_linear(envelope)
        rhythm_feat = rhythm_feat.view(batch_size, seq_len, 8).mean(dim=1) 

        # --- 步骤 5: 特征融合与 Logits 输出 ---
        combined_feat = torch.cat([weighted_feat, rhythm_feat], dim=1) 
        logits = self.warning_head(combined_feat)

        # --- 步骤 6: 实时生成 1D-CAM 可解释热力图 ---
        # 修正：切片提取 warning_head 中对应 TCN 通道的前 64 维权重
        cam_weights = self.warning_head[1].weight[:, :64].mean(dim=0)
        cam_1d = torch.relu(torch.sum(cam_weights.view(1, 64, 1) * global_skip, dim=1))
        # 归一化至 [0, 1] 区间以便前端渲染背景色深度
        cam_1d = cam_1d / (cam_1d.max(dim=1, keepdim=True)[0] + 1e-8)

        return {
            "logits": logits,                  # 训练用
            "prob": torch.sigmoid(logits),     # 业务预警用
            "cam": cam_1d,                     # 解释性前端渲染用 (长度 300)
            "attn": attn_weights.squeeze(1)    # 权重分析用
        }