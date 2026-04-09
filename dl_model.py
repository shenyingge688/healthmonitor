import torch
import torch.nn as nn

class WaveNetEdgeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.3):
        super(WaveNetEdgeBlock, self).__init__()
        self.pad_len = (kernel_size - 1) * dilation
        self.pad = nn.ConstantPad1d((self.pad_len, 0), 0)
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout1d(dropout) 
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout1d(dropout)
        
        self.shape_match = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else nn.Identity()
            
        self.final_relu = nn.ReLU()

    def forward(self, x):
        res = self.relu1(self.bn1(self.conv1(self.pad(x))))
        res = self.drop1(res)
        res = self.relu2(self.bn2(self.conv2(self.pad(res))))
        res = self.drop2(res)
        return self.final_relu(self.shape_match(x) + res), res

class HybridWarningNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super(HybridWarningNet, self).__init__()
        
        self.cnn_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 8, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(8), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16), nn.ReLU(),
            # 🟢 关键修改：改为最大池化，像雷达一样精准提取并保留最异常的波段！
            nn.AdaptiveMaxPool1d(1) 
        )
        
        self.tcn_blocks = nn.ModuleList([
            WaveNetEdgeBlock(16, 32, kernel_size=5, dilation=1),
            WaveNetEdgeBlock(32, 32, kernel_size=5, dilation=2),
            WaveNetEdgeBlock(32, 32, kernel_size=5, dilation=4),
            WaveNetEdgeBlock(32, 64, kernel_size=5, dilation=8)
        ])
        
        self.skip_adapters = nn.ModuleList([nn.Conv1d(32, 64, 1) for _ in range(3)])
        
        self.warning_head = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(64, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid() 
        )

    def forward(self, x):
        batch_size, seq_len, channels, points = x.size()
        
        c_in = x.view(batch_size * seq_len, channels, points)
        c_out = self.cnn_extractor(c_in).squeeze(-1) 
        t_in = c_out.view(batch_size, seq_len, 16).transpose(1, 2)
        
        global_skip = 0
        for i, block in enumerate(self.tcn_blocks):
            t_in, skip_out = block(t_in)
            if i < len(self.skip_adapters):
                global_skip += self.skip_adapters[i](skip_out)
            else:
                global_skip += skip_out
            
        return None, self.warning_head(global_skip[:, :, -1])