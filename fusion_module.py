# Transformer 融合模块
# 使用轻量化 MobileViT 实现特征融合
# 针对场景：高清背景图 + 低光含人图，考虑位置偏移

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class MobileViTAttention(nn.Module):
    """
    轻量化 MobileViT 注意力模块
    针对场景优化：
    1. 高清背景图可能和低光图有轻微位置偏移
    2. 高清图无目标，低光图包含行人
    3. 只检测行人，需要保护行人特征不被背景稀释
    """
    
    def __init__(self):
        super(MobileViTAttention, self).__init__()
        
        # 特征维度配置
        self.feature_dim = Config.FEATURE_DIM
        
        # 预初始化投影层
        self.high_res_proj = nn.Conv2d(128, 16, kernel_size=1)
        self.output_proj = nn.Conv2d(32, 16, kernel_size=1)
        
        # 新增：轻量级配准网络（估计位置偏移）
        # 使用 3 层卷积估计光流场
        self.aligner = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=3, padding=1)
        )
        
        # 新增：人体区域估计网络（不依赖检测器）
        # 原理：人体区域纹理复杂，特征响应强
        self.human_estimator = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, low_light_feat, high_res_feat, dark_mask):
        """
        前向传播 - 改进的融合方法
        
        Args:
            low_light_feat: 低光帧特征 (B, C, H, W)
            high_res_feat: 高清缓存特征 (B, 128) 或 (B, 128, H, W)
            dark_mask: 暗部掩码 (B, 1, H, W)
        
        Returns:
            融合后的特征 (B, C, H, W)
        """
        B, C, H, W = low_light_feat.shape
        
        # ==================== Step 1: 维度适配 ====================
        if high_res_feat.dim() == 2:
            # 全局特征 (B, 128) → 空间特征 (B, 128, H, W)
            high_res_feat = high_res_feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        
        # 高清特征投影：128 通道 → 16 通道
        high_res_proj = self.high_res_proj(high_res_feat)
        
        # ==================== Step 2: 图像配准（对齐偏移） ====================
        # 估计低光特征和高清特征之间的偏移
        concat_for_flow = torch.cat([low_light_feat, high_res_proj], dim=1)
        flow = self.aligner(concat_for_flow)  # (B, 2, H, W)
        
        # 生成采样网格
        grid = self._create_grid(B, H, W, low_light_feat.device)
        flow_grid = grid + flow
        flow_grid = flow_grid.permute(0, 2, 3, 1)
        flow_grid[..., 0] = flow_grid[..., 0] / (W - 1) * 2 - 1
        flow_grid[..., 1] = flow_grid[..., 1] / (H - 1) * 2 - 1
        
        # 双线性采样，对齐高清特征
        # MPS 不支持 'border' padding_mode，使用 'zeros' 替代
        padding_mode = 'zeros' if low_light_feat.device.type == 'mps' else 'border'
        aligned_high_res = F.grid_sample(
            high_res_proj, 
            flow_grid, 
            mode='bilinear', 
            padding_mode=padding_mode
        )
        
        # ==================== Step 3: 估计人体区域 ====================
        # 使用低光特征估计人体区域
        # 原理：人体区域纹理复杂，特征响应强
        human_response = torch.max(low_light_feat, dim=1, keepdim=True)[0]
        human_response = F.sigmoid(human_response * 10 - 3)
        
        # 使用估计网络进一步细化
        human_map = self.human_estimator(low_light_feat)
        
        # 结合两种响应
        # human_map: 基于学习的估计
        # human_response: 基于统计的估计
        human_map = human_map * 0.7 + human_response * 0.3
        
        # ==================== Step 4: 计算融合权重 ====================
        # 关键创新：三个维度的权重
        # 1. 人体权重：人体区域不融合背景
        # 2. 暗部权重：只在暗部融合
        # 3. 质量权重：特征差异大的区域减少融合
        
        # 人体权重（人体区域保护）
        human_weight = human_map  # 人体区域值大
        
        # 暗部权重（只在暗部融合）
        dark_weight = dark_mask.float()
        
        # 特征差异权重（差异大说明可能是人体）
        feat_diff = torch.abs(low_light_feat - aligned_high_res).mean(dim=1, keepdim=True)
        diff_weight = F.sigmoid(feat_diff * 5 - 1)  # 差异大 → 权重小
        
        # 综合权重
        # 人体区域：不融合 (weight=0)
        # 暗部背景区域：融合 (weight=1)
        # 亮部区域：不融合 (weight=0)
        fuse_weight = (1 - human_weight) * dark_weight * (1 - diff_weight)
        fuse_weight = fuse_weight.clamp(0, 1)
        
        # ==================== Step 5: 加权融合 ====================
        # 残差融合策略
        residual = aligned_high_res - low_light_feat
        
        # 只在背景暗部区域应用残差
        fused_feat = low_light_feat + residual * fuse_weight
        
        # ==================== Step 6: 特征整合 ====================
        combined = torch.cat([low_light_feat, fused_feat], dim=1)
        fused_feat = self.output_proj(combined)
        
        return fused_feat
    
    def _create_grid(self, B, H, W, device):
        """创建采样网格"""
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).float()
        return grid.unsqueeze(0).repeat(B, 1, 1, 1)
    
    def get_dark_mask(self, image):
        """
        生成暗部掩码 - 判断图像中的暗部区域
        
        Args:
            image: 输入图像
               可能是单张图像 (H, W, 3) - channels-last 格式
               或批量图像 (B, 3, H, W) - channels-first 格式
        
        Returns:
            暗部掩码
            单张图像：(1, H, W)
            批量图像：(B, 1, H, W)
            掩码值：1 表示暗部，0 表示亮部
        """
        if image.dim() == 3:
            # 单张图像 (H, W, 3)
            gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
            mask = (gray < Config.DARK_THRESHOLD).float().unsqueeze(0)
        else:
            # 批量图像 (B, 3, H, W)
            gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
            mask = (gray < Config.DARK_THRESHOLD).float().unsqueeze(1)
        
        return mask
