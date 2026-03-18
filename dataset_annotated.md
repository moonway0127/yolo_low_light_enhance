# dataset.py 完整注释

```python
# 数据集模块
# 加载高清图 + 低光帧配对数据集
# 原理：构建 PyTorch Dataset，支持数据增强和自动标注

import os  # 操作系统接口
import cv2  # OpenCV 计算机视觉库
import numpy as np  # 数值计算库
import torch  # PyTorch 框架
import torch.nn as nn  # 神经网络模块
from torch.utils.data import Dataset  # PyTorch 数据集基类
from ultralytics import YOLO  # YOLO 目标检测模型
from config import Config  # 配置参数

class ZeroDCE(nn.Module):
    """
    简化版 Zero-DCE 模型
    用于从高清图生成低光图 (训练数据准备)
    
    核心原理:
    - Zero-DCE: Zero-Reference Deep Curve Estimation
    - 将低光增强视为像素级曲线估计问题
    - 这里反向使用，生成低光图像
    """
    
    def __init__(self):
        """初始化 Zero-DCE 网络结构"""
        # 调用父类构造函数
        super(ZeroDCE, self).__init__()
        
        # ReLU 激活函数
        self.relu = nn.ReLU()
        
        # 卷积层配置 - 简化版，只有 4 层
        # 第 1 层：3 通道 (RGB) → 32 通道
        # kernel_size=3: 3x3 卷积核
        # stride=1: 步长为 1
        # padding=1: 填充 1 像素，保持尺寸不变
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        
        # 第 2 层：32 → 32 通道
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        # 第 3 层：32 → 32 通道
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        # 第 4 层：32 → 3 通道 (RGB 输出)
        self.conv4 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        """
        前向传播 - 生成低光图像
        
        Args:
            x: 输入高清图 (B, 3, H, W), 范围 [0, 255]
        
        Returns:
            生成的低光图 (B, 3, H, W), 范围 [0, 255]
        """
        # 输入归一化到 [0, 1]
        x = x / 255.0
        
        # 特征提取 - 卷积层 + ReLU 激活
        out = self.relu(self.conv1(x))  # 第 1 层
        out = self.relu(self.conv2(out))  # 第 2 层
        out = self.relu(self.conv3(out))  # 第 3 层
        out = self.conv4(out)  # 第 4 层 (无激活)
        
        # 生成增强映射 (sigmoid → [0, 1])
        out = torch.sigmoid(out)
        
        # 应用逆映射 - 生成低光图
        # 原理：low_light = original * (1 - enhancement_map)
        # 与增强操作相反，这里降低亮度
        low_light = x * (1 - out)
        
        # 裁剪到 [0, 1] 范围
        low_light = torch.clamp(low_light, 0, 1)
        
        # 转换回 [0, 255] 范围
        low_light = low_light * 255.0
        
        return low_light

class LowLightPairDataset(Dataset):
    """
    低光照配对数据集 (继承自 PyTorch Dataset)
    用于加载高清图和低光图的配对数据
    
    核心功能:
    1. 加载图像对 (高清 + 低光)
    2. 自动生成低光图 (如果没有真实低光图)
    3. 数据增强 (翻转、裁剪、噪声等)
    4. 自动标注 (使用 YOLO 生成检测标签)
    """
    
    def __init__(self, high_res_dir, low_light_dir=None, transform=None, 
                 auto_label=True, model_path='yolo26n.pt', generate_low_light=True):
        """
        初始化数据集
        
        Args:
            high_res_dir: 高清图像目录路径
            low_light_dir: 低光图像目录路径 (可选)
            transform: 数据增强变换函数 (未使用)
            auto_label: 是否启用自动标注
            model_path: YOLO 模型路径 (用于自动标注)
            generate_low_light: 是否自动生成低光图像
        """
        # 保存目录路径
        self.high_res_dir = high_res_dir
        self.low_light_dir = low_light_dir
        
        # 保存变换函数
        self.transform = transform
        
        # 配置自动标注
        self.auto_label = auto_label
        
        # 配置低光图生成
        self.use_generate_low_light = generate_low_light
        
        # 初始化 YOLO 模型 (用于自动标注)
        # 条件：如果启用自动标注
        if auto_label:
            self.yolo_model = YOLO(model_path)
        
        # 初始化 Zero-DCE 模型 (用于生成低光图)
        # 条件：如果启用低光图生成
        if generate_low_light:
            self.zero_dce = self._build_zero_dce()
        
        # 获取高清图文件列表
        self.high_res_files = self._get_image_files(high_res_dir)
        
        # 处理低光图目录
        if low_light_dir and os.path.exists(low_light_dir):
            # 低光目录存在
            self.low_light_files = self._get_image_files(low_light_dir)
            
            # 尝试匹配图像对
            if len(self.low_light_files) > 0:
                # 有低光图像文件
                self.pair_files = self._match_files()
                
                # 检查匹配结果
                if len(self.pair_files) == 0:
                    # 没有匹配成功，使用高清图生成低光图
                    self.pair_files = [(f, None) for f in self.high_res_files]
                    print("未找到匹配的图像对，使用高光图像生成低光图像")
            else:
                # 低光目录为空，使用高清图生成
                self.pair_files = [(f, None) for f in self.high_res_files]
                print("低光目录中没有图像文件，使用高光图像生成低光图像")
        else:
            # 低光目录不存在，使用高清图生成
            self.pair_files = [(f, None) for f in self.high_res_files]
            print("低光目录不存在，使用高光图像生成低光图像")
        
        # 打印数据集信息
        print(f"数据集初始化完成，共 {len(self.pair_files)} 对图像")
    
    def _get_image_files(self, directory):
        """
        获取目录中的所有图像文件
        
        Args:
            directory: 目录路径
        
        Returns:
            图像文件路径的排序列表
        """
        # 支持的图像格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # 文件列表
        files = []
        
        # 检查目录是否存在
        if not os.path.exists(directory):
            print(f"警告：目录 {directory} 不存在")
            return files
        
        # 遍历目录
        try:
            for file in os.listdir(directory):
                # 检查文件扩展名 (不区分大小写)
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    # 添加完整路径
                    files.append(os.path.join(directory, file))
            
            # 打印找到的文件数
            print(f"在目录 {directory} 中找到 {len(files)} 个图像文件")
        except Exception as e:
            # 错误处理
            print(f"读取目录 {directory} 时出错：{e}")
        
        # 返回排序后的文件列表
        return sorted(files)
    
    def _match_files(self):
        """
        匹配高清图和低光图 (基于文件名)
        
        Returns:
            配对的 (高清图路径，低光图路径) 列表
        """
        # 配对列表
        pair_files = []
        
        # 遍历高清图文件
        for high_res_file in self.high_res_files:
            # 获取文件名 (不含路径)
            filename = os.path.basename(high_res_file)
            
            # 构造低光图路径
            low_light_file = os.path.join(self.low_light_dir, filename)
            
            # 检查低光图是否存在
            if low_light_file in self.low_light_files:
                # 找到匹配，添加到配对列表
                pair_files.append((high_res_file, low_light_file))
        
        # 检查匹配结果
        if len(pair_files) == 0:
            print("警告：未找到匹配的图像对")
        
        return pair_files
    
    def _build_zero_dce(self):
        """
        构建 Zero-DCE 模型
        
        Returns:
            ZeroDCE 模型实例
        """
        return ZeroDCE()
    
    def generate_low_light(self, high_res):
        """
        使用 Zero-DCE 生成低光图像
        
        Args:
            high_res: 高清图 (H, W, 3), numpy 数组，范围 [0, 255]
        
        Returns:
            生成的低光图 (H, W, 3), numpy 数组，范围 [0, 255]
        """
        # 检查是否启用生成
        if not self.use_generate_low_light:
            return high_res
        
        # 转换为 PyTorch 张量
        # .permute(2, 0, 1): (H, W, C) → (C, H, W)
        # .unsqueeze(0): 增加 batch 维度 (1, C, H, W)
        img_tensor = torch.from_numpy(high_res).permute(2, 0, 1).unsqueeze(0).float()
        
        # 前向传播 (禁用梯度)
        with torch.no_grad():
            low_light_tensor = self.zero_dce(img_tensor)
        
        # 转换回 numpy 数组
        # .squeeze(0): 移除 batch 维度
        # .permute(1, 2, 0): (C, H, W) → (H, W, C)
        low_light = low_light_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype('uint8')
        
        return low_light
    
    def _apply_augmentation(self, high_res, low_light):
        """
        应用数据增强
        
        Args:
            high_res: 高清图 (H, W, 3)
            low_light: 低光图 (H, W, 3)
        
        Returns:
            增强后的 (high_res, low_light)
        """
        # 1. 随机水平翻转 (50% 概率)
        if np.random.random() > 0.5:
            high_res = cv2.flip(high_res, 1)  # 1 表示水平翻转
            low_light = cv2.flip(low_light, 1)
        
        # 2. 随机裁剪 (50% 概率)
        if np.random.random() > 0.5:
            h, w = high_res.shape[:2]  # 获取高宽
            crop_size = int(min(h, w) * 0.8)  # 裁剪到 80%
            x = np.random.randint(0, w - crop_size)  # 随机 x 坐标
            y = np.random.randint(0, h - crop_size)  # 随机 y 坐标
            
            high_res = high_res[y:y+crop_size, x:x+crop_size]
            low_light = low_light[y:y+crop_size, x:x+crop_size]
        
        # 3. 亮度扰动 (50% 概率)
        if np.random.random() > 0.5:
            # 随机亮度因子 [0.8, 1.2]
            brightness_factor = 0.8 + np.random.random() * 0.4
            # 应用亮度调整
            high_res = np.clip(high_res * brightness_factor, 0, 255).astype(np.uint8)
            low_light = np.clip(low_light * brightness_factor, 0, 255).astype(np.uint8)
        
        # 4. 高斯噪声 (50% 概率)
        if np.random.random() > 0.5:
            # 生成高斯噪声 (均值 0, 标准差 10)
            noise = np.random.normal(0, 10, low_light.shape).astype(np.uint8)
            # 添加噪声
            low_light = np.clip(low_light + noise, 0, 255).astype(np.uint8)
        
        # 5. 调整大小到输入尺寸
        high_res = cv2.resize(high_res, (Config.INPUT_SIZE, Config.INPUT_SIZE))
        low_light = cv2.resize(low_light, (Config.INPUT_SIZE, Config.INPUT_SIZE))
        
        return high_res, low_light
    
    def _auto_label(self, image):
        """
        使用 YOLO 模型自动标注图像
        
        Args:
            image: 输入图像 (H, W, 3)
        
        Returns:
            标注列表 [[cls, x, y, w, h], ...]
            cls: 类别 ID
            x, y: 中心坐标 (归一化)
            w, h: 宽高 (归一化)
        """
        # 检查是否启用自动标注
        if not self.auto_label:
            return []
        
        # YOLO 检测
        results = self.yolo_model(image, verbose=False)
        
        # 标注列表
        labels = []
        
        # 处理检测结果
        for result in results:
            # 检查是否有检测框
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    # 获取边界框 (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # 获取类别
                    cls = int(box.cls[0].item())
                    
                    # 获取图像尺寸
                    h, w = image.shape[:2]
                    
                    # 转换为归一化的中心坐标和宽高
                    center_x = (x1 + x2) / 2 / w
                    center_y = (y1 + y2) / 2 / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    
                    # 确保坐标在 [0, 1] 范围内
                    center_x = max(0, min(1, center_x))
                    center_y = max(0, min(1, center_y))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))
                    
                    # 添加到标注列表
                    labels.append([cls, center_x, center_y, width, height])
        
        return labels
    
    def __getitem__(self, idx):
        """
        获取数据项 (PyTorch Dataset 的必需方法)
        
        Args:
            idx: 索引
        
        Returns:
            (high_res_tensor, low_light_tensor, labels)
        """
        # 获取文件路径对
        high_res_path, low_light_path = self.pair_files[idx]
        
        # 打印进度
        print(f"处理图像 {idx+1}/{len(self.pair_files)}: {os.path.basename(high_res_path)}")
        
        # 加载图像
        try:
            # 加载高清图
            high_res = cv2.imread(high_res_path)
            
            # 加载或生成低光图
            if low_light_path:
                low_light = cv2.imread(low_light_path)
                if low_light is None:
                    # 加载失败，生成低光图
                    print(f"生成低光图像：{os.path.basename(high_res_path)}")
                    low_light = self.generate_low_light(high_res)
            else:
                # 没有低光图路径，生成低光图
                print(f"生成低光图像：{os.path.basename(high_res_path)}")
                low_light = self.generate_low_light(high_res)
            
            # 检查高清图加载
            if high_res is None:
                print(f"警告：无法加载高清图像 {high_res_path}")
                # 返回占位符
                high_res = np.zeros((Config.INPUT_SIZE, Config.INPUT_SIZE, 3), dtype=np.uint8)
                low_light = np.zeros((Config.INPUT_SIZE, Config.INPUT_SIZE, 3), dtype=np.uint8)
                labels = []
            else:
                # 应用数据增强
                high_res, low_light = self._apply_augmentation(high_res, low_light)
                
                # 自动标注
                labels = self._auto_label(high_res)
                print(f"标注完成，检测到 {len(labels)} 个目标")
            
            # 转换为 PyTorch 张量
            # .permute(2, 0, 1): (H, W, C) → (C, H, W)
            high_res = torch.from_numpy(high_res).permute(2, 0, 1).float()
            low_light = torch.from_numpy(low_light).permute(2, 0, 1).float()
            
            return high_res, low_light, labels
        except Exception as e:
            # 错误处理
            print(f"加载图像时出错：{e}")
            # 返回占位符
            high_res = torch.zeros(3, Config.INPUT_SIZE, Config.INPUT_SIZE)
            low_light = torch.zeros(3, Config.INPUT_SIZE, Config.INPUT_SIZE)
            return high_res, low_light, []
    
    def __len__(self):
        """
        获取数据集大小 (PyTorch Dataset 的必需方法)
        
        Returns:
            数据集大小 (图像对数量)
        """
        return len(self.pair_files)
```
