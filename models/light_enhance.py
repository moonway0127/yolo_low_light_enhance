# 低光增强模块
# 使用轻量化 Zero-DCE 实现低光图像增强

import torch
import torch.nn as nn
import torch.nn.functional as F

class LightEnhance(nn.Module):
    """
    低光增强类（轻量化 Zero-DCE）
    用于快速增强低光图像的亮度和对比度
    """
    
    def __init__(self):
        """
        初始化低光增强模型
        """
        super(LightEnhance, self).__init__()
        
        # 轻量化网络结构
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        
        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入低光帧 (B, 3, H, W)
            
        Returns:
            增强后的帧 (B, 3, H, W)
        """
        # 输入归一化
        x = x / 255.0 if x.max() > 1.0 else x
        
        # 特征提取
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        
        # 生成增强映射
        enhancement_map = self.sigmoid(self.conv5(out))
        
        # 应用增强映射
        enhanced = x * (1 + enhancement_map)
        
        # 确保输出在 [0, 1] 范围内
        enhanced = torch.clamp(enhanced, 0, 1)
        
        return enhanced
    
    def enhance_frame(self, frame):
        """
        增强单帧图像
        
        Args:
            frame: 输入低光帧 (H, W, 3)，范围 [0, 255]
            
        Returns:
            增强后的帧 (H, W, 3)，范围 [0, 255]
        """
        # 转换为张量
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
        
        # 前向传播
        with torch.no_grad():
            enhanced = self.forward(frame_tensor)
        
        # 转换回 numpy 数组
        enhanced_frame = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy().astype('uint8')
        
        return enhanced_frame
