"""
HealthMonitor V3.0 - 深度学习网络模型主体架构
功能描述：网络聚合了时间序列因果约束、感受野扩大、形态特征与长程规律并行提取得架构体系。
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    """
    因果裁剪组件：
    在使用特征填充（Padding）的情况下保障 TCN 节点无法触及未来时序上的特征信息。
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """
    带旁路透传设计的扩张因果卷积块 (Dilated Causal Convolution)
    功能：内部实现双层残差结构，并对外抛出主干和跳跃连接(Skip Connection)两股数据流。
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)  # <--- 【减震器 1】
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)  # <--- 【减震器 2】
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 把 bn1 和 bn2 组装进执行链路中（紧跟在 chomp 之后，relu 之前）
        self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2)
        
        # 残差对齐映射通道
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        main_out = self.relu(out + res)
        skip_out = out
        return main_out, skip_out

class HybridWarningNet(nn.Module):
    """
    多模态时序聚合网络骨架。
    集成了 CNN 特征分支、长程 TCN 主干，并且引出了线性的1D类激活映射(CAM)。
    """
    def __init__(self, in_channels=1):
        super().__init__()
        
        # 形态提取段，聚焦极微观的QRS等波群
        self.cnn_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=15, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 长时序关联段，扩大捕捉区间范围
        num_channels = [32, 64, 64]
        kernel_size = 3
        layers = []
        in_ch = 32
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            out_ch = num_channels[i]
            layers.append(TemporalBlock(
                in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size
            ))
            in_ch = out_ch
        self.tcn_blocks = nn.ModuleList(layers)
        
        self.skip_adapters = nn.ModuleList([
            nn.Conv1d(32, 64, 1),
            nn.Conv1d(64, 64, 1)
        ])
        
        # 注意力层用以在长距离上赋予不同病灶权重分布
        self.temporal_attn = nn.Sequential(
            nn.Conv1d(64, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 辅佐的节律物理池化通道
        self.rhythm_pool = nn.AvgPool1d(kernel_size=25, stride=25)
        self.rhythm_linear = nn.Linear(20, 8)
        
        # 汇总决策归类器
        self.warning_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(72, 16),
            nn.ReLU(),
            nn.Linear(16, 6)
        )

    def forward(self, x):
        batch_size, seq_len, channels, points = x.size()
        c_in = x.view(batch_size * seq_len, channels, points)
        
        cnn_features = self.cnn_extractor(c_in)
        c_out = torch.cat([self.max_pool(cnn_features).squeeze(-1), self.avg_pool(cnn_features).squeeze(-1)], dim=1)
        
        t_in = c_out.view(batch_size, seq_len, 32).transpose(1, 2)
        global_skip = 0
        for i, block in enumerate(self.tcn_blocks):
            t_in, skip_out = block(t_in)
            global_skip += self.skip_adapters[i](skip_out) if i < len(self.skip_adapters) else skip_out
            
        attn_weights = self.temporal_attn(global_skip)
        weighted_feat = torch.sum(global_skip * attn_weights, dim=2) / (attn_weights.sum(dim=2) + 1e-4)
        
        envelope = self.rhythm_pool(torch.abs(c_in)).view(batch_size * seq_len, -1)
        rhythm_feat = self.rhythm_linear(envelope).view(batch_size, seq_len, 8).mean(dim=1)
        
        combined_feat = torch.cat([weighted_feat, rhythm_feat], dim=1)
        logits = self.warning_head(combined_feat)
        
        # 输出线性权重映射特征矩阵(1D-CAM热力图用)
        cam_weights = self.warning_head[1].weight[:, :64].mean(dim=0)
        cam_1d = torch.relu(torch.sum(cam_weights.view(1, 64, 1) * global_skip, dim=1))
        cam_1d = cam_1d / (cam_1d.max(dim=1, keepdim=True)[0] + 1e-8)
        
        return {
            "logits": logits,
            "prob": torch.softmax(logits, dim=1),
            "cam": cam_1d,
            "attn": attn_weights.squeeze(1)
        }