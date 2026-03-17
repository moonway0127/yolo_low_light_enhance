# Transformer 融合模块
# 使用轻量化 MobileViT 实现特征融合

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class MobileViTAttention(nn.Module):
    """
    轻量化 MobileViT 注意力模块
    用于融合低光帧特征和高清缓存特征
    """
    
    def __init__(self):
        """
        初始化 MobileViT 注意力模块
        """
        super(MobileViTAttention, self).__init__()
        
        # 特征维度
        self.feature_dim = Config.FEATURE_DIM
        
        # 预初始化投影层（避免动态创建）
        self.high_res_proj = nn.Conv2d(128, 16, kernel_size=1)
        self.output_proj = nn.Conv2d(32, 16, kernel_size=1)
    
    def forward(self, low_light_feat, high_res_feat, dark_mask):
        """
        前向传播
        
        Args:
            low_light_feat: 低光帧特征 (B, C, H, W)
            high_res_feat: 高清缓存特征 (B, 128) 或 (B, 128, H, W)
            dark_mask: 暗部掩码 (B, 1, H, W)
            
        Returns:
            融合后的特征 (B, C, H, W)
        """
        B, C, H, W = low_light_feat.shape
        
        # 处理高清特征
        if high_res_feat.dim() == 2:
            # 如果是全局特征，扩展到空间维度
            # 使用 expand 而不是 repeat，减少内存使用
            high_res_feat = high_res_feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        
        # 维度适配（使用预初始化的层）
        high_res_feat_proj = self.high_res_proj(high_res_feat)
        
        # 简单的特征融合：拼接后卷积
        combined = torch.cat([low_light_feat, high_res_feat_proj], dim=1)
        fused_feat = self.output_proj(combined)
        
        # 仅对暗部区域应用融合
        dark_mask = dark_mask.float()
        fused_feat = low_light_feat * (1 - dark_mask) + (low_light_feat + fused_feat) * dark_mask
        
        return fused_feat
    
    def get_dark_mask(self, image):
        """
        生成暗部掩码
        
        Args:
            image: 输入图像 (B, 3, H, W) 或 (H, W, 3)
            
        Returns:
            暗部掩码 (B, 1, H, W) 或 (1, H, W)
        """
        if image.dim() == 3:
            # 单张图像
            gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
            mask = (gray < Config.DARK_THRESHOLD).float().unsqueeze(0)
        else:
            # 批量图像
            gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
            mask = (gray < Config.DARK_THRESHOLD).float().unsqueeze(1)
        
        return mask
