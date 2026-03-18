# Transformer 融合模块
# 使用轻量化 MobileViT 实现特征融合
# 原理：将低光特征与高清缓存特征融合，只在暗部区域应用增强

import torch  # 导入 PyTorch 深度学习框架
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式接口
from config import Config  # 导入配置模块，获取特征维度等参数

class MobileViTAttention(nn.Module):
    """
    轻量化 MobileViT 注意力模块
    用于融合低光帧特征和高清缓存特征
    
    核心原理:
    - 受 MobileViT 启发的轻量级注意力机制
    - 使用 1x1 卷积进行特征投影和融合
    - 基于暗部掩码的自适应加权融合
    - 只在需要的区域 (暗部) 应用高清特征，节省计算资源
    """
    
    def __init__(self):
        """
        初始化 MobileViT 注意力模块
        构建特征投影层，用于维度适配和特征融合
        """
        # 调用父类 nn.Module 的构造函数
        super(MobileViTAttention, self).__init__()
        
        # 特征维度配置
        # 从 Config 类获取 FEATURE_DIM (128)
        # 用于确保高清特征的维度一致性
        self.feature_dim = Config.FEATURE_DIM
        
        # 预初始化投影层 - 避免在 forward 中动态创建层
        # 原因：动态创建的层不会被优化器跟踪，无法训练
        
        # 高清特征投影层：1x1 卷积
        # 输入：128 通道 (高清全局特征)
        # 输出：16 通道 (与低光特征通道数匹配)
        # kernel_size=1: 1x1 卷积，只调整通道数，不改变空间尺寸
        # 作用：将高维高清特征投影到低维空间，便于融合
        self.high_res_proj = nn.Conv2d(128, 16, kernel_size=1)
        
        # 输出投影层：1x1 卷积
        # 输入：32 通道 (低光特征 16 + 高清投影特征 16)
        # 输出：16 通道 (融合后的特征)
        # 作用：融合拼接后的特征，学习特征间的交互
        self.output_proj = nn.Conv2d(32, 16, kernel_size=1)
    
    def forward(self, low_light_feat, high_res_feat, dark_mask):
        """
        前向传播 - 特征融合的核心计算
        
        Args:
            low_light_feat: 低光帧特征 (B, C, H, W)
               B: batch size
               C: 通道数 (通常是 16)
               H: 特征图高度
               W: 特征图宽度
            
            high_res_feat: 高清缓存特征
               可能是全局特征 (B, 128) - 1x1 空间尺寸
               或空间特征 (B, 128, H, W) - 与低光特征同尺寸
            
            dark_mask: 暗部掩码 (B, 1, H, W)
               二值掩码，1 表示暗部区域，0 表示亮部区域
            
        Returns:
            融合后的特征 (B, C, H, W)
            与输入低光特征相同的形状
        """
        # 获取低光特征的形状信息
        # B: batch size (批次大小)
        # C: channels (通道数)
        # H: height (高度)
        # W: width (宽度)
        B, C, H, W = low_light_feat.shape
        
        # 处理高清特征 - 维度适配
        # 检查高清特征的维度数
        if high_res_feat.dim() == 2:
            # 情况 1：高清特征是全局特征 (B, 128)
            # 需要扩展到空间维度，以便与低光特征融合
            
            # .unsqueeze(-1): 在最后增加一个维度
            #   (B, 128) -> (B, 128, 1)
            # .unsqueeze(-1): 再增加一个维度
            #   (B, 128, 1) -> (B, 128, 1, 1)
            # .expand(-1, -1, H, W): 扩展到目标尺寸
            #   (B, 128, 1, 1) -> (B, 128, H, W)
            # -1 表示保持该维度不变
            # 使用 expand 而不是 repeat 的原因：
            # - expand 不占用额外内存，只是视图操作
            # - repeat 会复制数据，占用更多内存
            high_res_feat = high_res_feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        
        # 维度适配 - 高清特征投影
        # 使用预定义的 1x1 卷积层将高清特征从 128 通道投影到 16 通道
        # 作用：
        # 1. 调整通道数，使其与低光特征一致
        # 2. 学习特征变换，更好地融合
        high_res_feat_proj = self.high_res_proj(high_res_feat)
        
        # 特征融合 - 拼接 + 卷积
        # torch.cat: 沿指定维度拼接张量
        # dim=1: 沿通道维度拼接
        #   low_light_feat: (B, 16, H, W)
        #   high_res_feat_proj: (B, 16, H, W)
        #   combined: (B, 32, H, W)
        combined = torch.cat([low_light_feat, high_res_feat_proj], dim=1)
        
        # 融合特征投影 - 1x1 卷积
        # 将拼接后的 32 通道特征投影回 16 通道
        # 作用：
        # 1. 恢复原始通道数
        # 2. 学习低光和高清特征的交互
        # 3. 降维减少计算量
        fused_feat = self.output_proj(combined)
        
        # 暗部掩码类型转换
        # 将掩码转换为 float 类型，以便进行乘法运算
        # 原始掩码可能是 bool 或 int 类型
        dark_mask = dark_mask.float()
        
        # 自适应加权融合 - 核心创新点
        # 原理：只在暗部区域应用高清特征融合
        # 公式：fused = low_light * (1 - mask) + (low_light + fused_feat) * mask
        # 
        # 分情况讨论：
        # 1. 亮部区域 (mask=0):
        #    fused = low_light * 1 + (low_light + fused_feat) * 0
        #    fused = low_light
        #    保持原始低光特征，不使用融合
        #
        # 2. 暗部区域 (mask=1):
        #    fused = low_light * 0 + (low_light + fused_feat) * 1
        #    fused = low_light + fused_feat
        #    应用融合增强，加入高清特征信息
        #
        # 优势：
        # - 亮部区域已经足够清晰，不需要增强
        # - 暗部区域需要额外信息，融合高清特征
        # - 自适应处理，提高效率和效果
        fused_feat = low_light_feat * (1 - dark_mask) + (low_light_feat + fused_feat) * dark_mask
        
        # 返回融合后的特征
        return fused_feat
    
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
        # 检查输入图像的维度数
        if image.dim() == 3:
            # 情况 1：单张图像 (H, W, 3)
            # OpenCV 格式，channels-last
            
            # RGB 转灰度图 - 加权平均
            # 使用标准灰度系数：
            # - R: 0.299 (红色贡献最小)
            # - G: 0.587 (绿色贡献最大，人眼对绿色敏感)
            # - B: 0.114 (蓝色贡献次小)
            # 原理：人眼对不同颜色敏感度不同
            gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
            
            # 生成二值掩码
            # gray < Config.DARK_THRESHOLD: 比较运算，返回 bool 张量
            # Config.DARK_THRESHOLD = 50: 暗部阈值
            # 灰度值 < 50 的区域被认为是暗部
            # .float(): 将 bool 转换为 float (True->1.0, False->0.0)
            # .unsqueeze(0): 增加 batch 维度
            #   (H, W) -> (1, H, W)
            mask = (gray < Config.DARK_THRESHOLD).float().unsqueeze(0)
        
        else:
            # 情况 2：批量图像 (B, 3, H, W)
            # PyTorch 格式，channels-first
            
            # RGB 转灰度图 - 批量处理
            # image[:, 0]: 所有图像的 R 通道 (B, H, W)
            # image[:, 1]: 所有图像的 G 通道 (B, H, W)
            # image[:, 2]: 所有图像的 B 通道 (B, H, W)
            gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
            
            # 生成二值掩码 - 批量处理
            # .unsqueeze(1): 增加通道维度
            #   (B, H, W) -> (B, 1, H, W)
            # 原因：掩码需要与特征图形状匹配，通道维度为 1
            mask = (gray < Config.DARK_THRESHOLD).float().unsqueeze(1)
        
        # 返回生成的暗部掩码
        return mask
