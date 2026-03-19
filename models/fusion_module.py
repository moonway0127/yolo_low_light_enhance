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
        
        # 新增：轻量级配准网络（估计位置偏移）- 金字塔结构
        # 粗配准层 (低分辨率)
        self.aligner_coarse = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=3, padding=1)
        )
        
        # 精配准层 (高分辨率)
        self.aligner_fine = nn.Sequential(
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
        
        # ==================== 可学习参数（） ====================
        # 1. 融合强度参数：控制整体融合强度
        self.fusion_strength = nn.Parameter(torch.tensor(1.0))
        
        # 2. 人体保护权重：控制人体区域的保护程度
        self.human_protection_weight = nn.Parameter(torch.tensor(0.7))
        
        # 3. 暗部融合权重：控制暗部区域的融合强度
        self.dark_fusion_weight = nn.Parameter(torch.tensor(1.0))
        
        # 4. 差异敏感度：控制对特征差异的敏感度
        self.diff_sensitivity = nn.Parameter(torch.tensor(5.0))
        
        # 5. 残差缩放因子：控制残差融合的比例
        self.residual_scale = nn.Parameter(torch.tensor(1.0))
        
        # 6. 双分支融合参数（类似图示模块）
        # 分支 1：增强型融合
        self.alpha = nn.Parameter(torch.tensor(0.5))
        # 分支 2：保守型融合
        self.beta = nn.Parameter(torch.tensor(0.3))
        
        # 7. 注意力门控机制：自适应选择融合策略
        # 输入是拼接后的特征 (16 + 16 = 32 通道)
        self.attention_gate = nn.Sequential(
            nn.Conv2d(32, 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 2, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 8. 通道注意力机制（新增）：自适应选择增强通道
        # 使用轻量级 SE-Block 风格
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局池化 (B, C, 1, 1)
            nn.Conv2d(16, 8, kernel_size=1),  # 降维
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=1),  # 升维
            nn.Sigmoid()  # 每个通道的权重 (B, 16, 1, 1)
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
        
        # ==================== Step 2: 图像配准（金字塔结构） ====================
        # 估计低光特征和高清特征之间的偏移
        concat_for_flow = torch.cat([low_light_feat, high_res_proj], dim=1)
        
        # 金字塔配准：粗配准 + 精配准
        # 1. 粗配准（下采样特征，估计大位移）
        concat_low = F.interpolate(concat_for_flow, scale_factor=0.5, mode='bilinear', align_corners=False)
        flow_coarse = self.aligner_coarse(concat_low)
        # 上采样回原始分辨率，并放大光流 (位移也放大 2 倍)
        # 使用原始特征尺寸，避免奇数尺寸问题
        flow_coarse = F.interpolate(flow_coarse, size=(H, W), mode='bilinear', align_corners=False) * 2.0
        
        # 2. 精配准（原始分辨率，估计小位移，叠加粗配准结果）
        flow_fine = self.aligner_fine(concat_for_flow)
        flow = flow_coarse + flow_fine  # 叠加两个光流场
        
        # 生成采样网格
        grid = self._create_grid(B, H, W, low_light_feat.device)
        flow_grid = grid + flow
        flow_grid = flow_grid.permute(0, 2, 3, 1)
        flow_grid[..., 0] = flow_grid[..., 0] / (W - 1) * 2 - 1
        flow_grid[..., 1] = flow_grid[..., 1] / (H - 1) * 2 - 1
        
        # 使用四次插值 (改进 2: 用 bilinear 替代，可能需要更多计算)
        # 保留更多高频细节
        padding_mode = 'zeros' if low_light_feat.device.type == 'mps' else 'border'
        aligned_high_res = F.grid_sample(
            high_res_proj, 
            flow_grid, 
            mode='bilinear', 
            padding_mode=padding_mode,
            align_corners=True  # 改进 3: 对齐角点
        )
        
        # 计算配准置信度 (改进 4: 评估配准质量)
        flow_magnitude = torch.sqrt(flow[:, 0:1, :, :] ** 2 + flow[:, 1:2, :, :] ** 2 + 1e-6)
        alignment_confidence = 1.0 - F.sigmoid(flow_magnitude * 2 - 1)  # 位移大 → 置信度低
        
        # ==================== Step 3: 估计人体区域 (改进版) ====================
        # 多线索融合的人体区域估计
        # 线索 1: 特征响应强度 (原方法)
        human_response = torch.max(low_light_feat, dim=1, keepdim=True)[0]
        human_response = F.sigmoid(human_response * 10 - 3)
        
        # 线索 2: 特征方差 (人体区域纹理复杂，方差大)
        feat_variance = torch.var(low_light_feat, dim=1, keepdim=True)
        feat_variance_norm = (feat_variance - feat_variance.min()) / (feat_variance.max() - feat_variance.min() + 1e-6)
        variance_response = F.sigmoid(feat_variance_norm * 5 - 2)
        
        # 线索 3: 多尺度特征 (使用不同通道子集)
        channel_groups = 4
        group_responses = []
        for i in range(channel_groups):
            start_ch = i * (C // channel_groups)
            end_ch = (i + 1) * (C // channel_groups)
            group_feat = low_light_feat[:, start_ch:end_ch, :, :]
            group_max = torch.max(group_feat, dim=1, keepdim=True)[0]
            group_response = F.sigmoid(group_max * 8 - 2)
            group_responses.append(group_response)
        multi_scale_response = torch.cat(group_responses, dim=1).mean(dim=1, keepdim=True)
        
        # 使用估计网络进一步细化
        human_map_learned = self.human_estimator(low_light_feat)
        
        # 多线索融合 (可学习权重)
        # human_map_learned: 基于学习的估计
        # human_response: 基于最大响应
        # variance_response: 基于方差
        # multi_scale_response: 多尺度响应
        human_map = (human_map_learned * self.human_protection_weight + 
                    human_response * 0.2 +
                    variance_response * 0.2 +
                    multi_scale_response * 0.2)
        
        # ==================== Step 4: 计算融合权重（可学习版本） ====================
        # 关键创新：三个维度的权重 + 可学习参数
        
        # 人体权重（人体区域保护）
        human_weight = human_map  # 人体区域值大
        
        # 暗部权重（只在暗部融合，使用可学习参数）
        dark_weight = dark_mask.float() * self.dark_fusion_weight
        
        # 特征差异权重（差异大说明可能是人体，使用可学习敏感度）
        feat_diff = torch.abs(low_light_feat - aligned_high_res).mean(dim=1, keepdim=True)
        diff_weight = F.sigmoid(feat_diff * self.diff_sensitivity - 1)  # 差异大 → 权重小
        
        # 综合权重
        # 人体区域：不融合 (weight=0)
        # 暗部背景区域：融合 (weight=1)
        # 亮部区域：不融合 (weight=0)
        fuse_weight = (1 - human_weight) * dark_weight * (1 - diff_weight)
        fuse_weight = fuse_weight.clamp(0, 1)
        
        # ==================== Step 5: 注意力门控双分支融合 ====================
        # 创新点：引入注意力门控，自适应选择融合策略
        
        # 计算注意力门控（学习如何融合）
        gate_input = torch.cat([low_light_feat, aligned_high_res], dim=1)
        attention_weights = self.attention_gate(gate_input)  # (B, 2, H, W)
        
        # 分支 1：增强型融合（更激进）
        residual_enhanced = aligned_high_res - low_light_feat
        fused_enhanced = low_light_feat + residual_enhanced * fuse_weight * self.alpha
        
        # 分支 2：保守型融合（更保守）
        residual_conservative = aligned_high_res - low_light_feat
        fused_conservative = low_light_feat + residual_conservative * fuse_weight * self.beta
        
        # 使用注意力权重自适应融合两个分支
        fused_feat = (fused_enhanced * attention_weights[:, 0:1] + 
                     fused_conservative * attention_weights[:, 1:2])
        
        # 应用全局融合强度和残差缩放
        fused_feat = low_light_feat + (fused_feat - low_light_feat) * self.fusion_strength * self.residual_scale
        
        # ==================== Step 5.5: 通道注意力加权 (改进版) ====================
        # 创新点：引入通道注意力，自适应选择增强通道
        channel_weights = self.channel_attention(fused_feat)  # (B, 16, 1, 1)
        
        # 添加细节保护：确保重要通道不被过度抑制
        channel_weights = torch.clamp(channel_weights, min=0.3)  # 最小权重 0.3，防止完全抑制
        
        fused_feat = fused_feat * channel_weights  # 每个通道乘以对应的权重
        
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
