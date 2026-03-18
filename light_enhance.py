# 低光增强模块
# 使用轻量化 Zero-DCE 实现低光图像增强
# 原理：基于 Zero-Reference Deep Curve Estimation 算法，通过学习曲线映射来增强图像

import torch  # 导入 PyTorch 深度学习框架
import torch.nn as nn  # 导入神经网络模块，提供层和模型构建功能
import torch.nn.functional as F  # 导入函数式接口，提供激活函数等操作

class LightEnhance(nn.Module):
    """
    低光增强类（轻量化 Zero-DCE）
    用于快速增强低光图像的亮度和对比度
    
    核心原理:
    - Zero-DCE 将低光增强视为像素级曲线估计问题
    - 通过调整像素值的曲线映射来增强亮度
    - 无需成对数据，可自监督学习
    """
    
    def __init__(self):
        """
        初始化低光增强模型
        构建轻量化 CNN 网络用于生成增强映射曲线
        """
        # 调用父类 nn.Module 的构造函数，初始化模块
        super(LightEnhance, self).__init__()
        
        # 轻量化网络结构 - 5 个卷积层
        # 设计理念：使用较少的层数和通道数，保证实时性
        
        # 第 1 个卷积层：输入 3 通道 (RGB) -> 输出 16 通道
        # kernel_size=3: 3x3 卷积核，捕捉局部特征
        # stride=1: 步长为 1，保持特征图尺寸不变
        # padding=1: 填充 1 像素，保持输出尺寸与输入一致
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        
        # 第 2 个卷积层：16 通道 -> 32 通道
        # 增加通道数，提取更丰富的特征
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # 第 3 个卷积层：32 通道 -> 32 通道
        # 保持通道数，进一步提取高级特征
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        # 第 4 个卷积层：32 通道 -> 16 通道
        # 减少通道数，为输出做准备
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        
        # 第 5 个卷积层：16 通道 -> 3 通道 (RGB)
        # 输出增强映射曲线，每个通道独立调整
        self.conv5 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        
        # 激活函数
        # ReLU: 线性整流函数，f(x) = max(0, x)，引入非线性
        self.relu = nn.ReLU()
        
        # Sigmoid: S 型函数，将输出压缩到 (0, 1) 范围
        # 用于生成增强映射曲线，控制每个像素的增强程度
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        前向传播 - 模型的核心计算流程
        
        Args:
            x: 输入低光帧 (B, 3, H, W)
               B: batch size (批次大小)
               3: RGB 三个通道
               H: 图像高度
               W: 图像宽度
            
        Returns:
            增强后的帧 (B, 3, H, W)，像素值范围 [0, 1]
        """
        # 输入归一化处理
        # 原理：神经网络通常在 [0, 1] 或 [-1, 1] 范围内训练，需要归一化
        # 判断：如果最大值>1.0，说明是 [0, 255] 范围，需要除以 255
        # 否则认为已经是 [0, 1] 范围，不需要处理
        x = x / 255.0 if x.max() > 1.0 else x
        
        # 特征提取阶段 - 通过卷积层逐层提取图像特征
        # 第 1 层卷积 + ReLU 激活
        # 作用：提取低级特征 (边缘、纹理等)
        out = self.relu(self.conv1(x))
        
        # 第 2 层卷积 + ReLU 激活
        # 作用：组合低级特征，形成中级特征
        out = self.relu(self.conv2(out))
        
        # 第 3 层卷积 + ReLU 激活
        # 作用：提取更抽象的特征表示
        out = self.relu(self.conv3(out))
        
        # 第 4 层卷积 + ReLU 激活
        # 作用：进一步提炼特征，为生成映射做准备
        out = self.relu(self.conv4(out))
        
        # 生成增强映射曲线
        # 原理：最后一个卷积层输出 3 通道，对应 RGB 每个通道的增强系数
        # sigmoid 激活：将输出压缩到 (0, 1) 范围
        # 物理意义：每个像素点的增强强度，值越大增强越多
        enhancement_map = self.sigmoid(self.conv5(out))
        
        # 应用增强映射到原始图像
        # 原理：enhanced = x * (1 + enhancement_map)
        # 解释：
        # - enhancement_map 范围是 (0, 1)
        # - (1 + enhancement_map) 范围是 (1, 2)
        # - 相当于将原始像素值提升 1-2 倍，实现亮度增强
        # - 每个通道独立增强，可以校正颜色
        enhanced = x * (1 + enhancement_map)
        
        # 输出值范围裁剪
        # 原理：增强后的值可能超过 1.0，需要限制在有效范围内
        # torch.clamp: 将张量中的值限制在 [min, max] 区间
        # 作用：防止像素值溢出，保证输出合法性
        enhanced = torch.clamp(enhanced, 0, 1)
        
        # 返回增强后的图像
        return enhanced
    
    def enhance_frame(self, frame):
        """
        增强单帧图像的便捷方法
        封装了张量转换和推理过程，直接处理 numpy 数组
        
        Args:
            frame: 输入低光帧 (H, W, 3)，范围 [0, 255]
                   H: 图像高度
                   W: 图像宽度
                   3: RGB 通道 (注意：OpenCV 是 BGR 顺序)
            
        Returns:
            增强后的帧 (H, W, 3)，范围 [0, 255]，uint8 类型
        """
        # 将 numpy 数组转换为 PyTorch 张量
        # torch.from_numpy(frame): 从 numpy 数组创建张量，共享内存
        # .permute(2, 0, 1): 调整维度顺序
        #   原始：(H, W, C) -> 转换后：(C, H, W)
        #   原因：PyTorch 使用 channels-first 格式，OpenCV 使用 channels-last
        # .unsqueeze(0): 增加 batch 维度
        #   (C, H, W) -> (1, C, H, W)
        #   原因：模型期望输入是 batch 形式
        # .float(): 转换为浮点型，模型需要 float32 类型
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
        
        # 前向传播 - 执行增强
        # with torch.no_grad(): 禁用梯度计算
        # 作用：
        # 1. 推理阶段不需要反向传播，禁用梯度可节省内存
        # 2. 加速计算，不需要构建计算图
        with torch.no_grad():
            enhanced = self.forward(frame_tensor)
        
        # 将增强后的张量转换回 numpy 数组
        # .squeeze(0): 移除 batch 维度
        #   (1, C, H, W) -> (C, H, W)
        # .permute(1, 2, 0): 恢复为 channels-last 格式
        #   (C, H, W) -> (H, W, C)
        # .cpu(): 将张量移到 CPU (如果在 GPU 上)
        # .numpy(): 转换为 numpy 数组
        # .astype('uint8'): 转换为 8 位无符号整数 (0-255)
        #   注意：这里直接转换可能会截断，应该先乘 255
        enhanced_frame = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy().astype('uint8')
        
        # 返回增强后的图像帧
        return enhanced_frame
