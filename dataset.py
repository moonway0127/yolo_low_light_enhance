# 数据集模块
# 加载高清图 + 低光帧配对数据集

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from ultralytics import YOLO
from config import Config

class ZeroDCE(nn.Module):
    def __init__(self):
        super(ZeroDCE, self).__init__()
        self.relu = nn.ReLU()
        # 简化版的Zero-DCE，只有几个卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        # 输入归一化
        x = x / 255.0
        
        # 前向传播
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.conv4(out)
        
        # 生成增强映射
        out = torch.sigmoid(out)
        
        # 应用增强映射（这里我们反向操作，生成低光图像）
        low_light = x * (1 - out)
        low_light = torch.clamp(low_light, 0, 1)
        
        # 转换回 [0, 255] 范围
        low_light = low_light * 255.0
        
        return low_light

class LowLightPairDataset(Dataset):
    """
    低光照配对数据集
    加载高清图和对应的低光帧
    """
    
    def __init__(self, high_res_dir, low_light_dir=None, transform=None, auto_label=True, model_path='yolo26n.pt', generate_low_light=True):
        """
        初始化数据集
        
        Args:
            high_res_dir: 高清图像目录
            low_light_dir: 低光图像目录（可选）
            transform: 数据增强变换
            auto_label: 是否自动标注
            model_path: 用于自动标注的YOLO模型路径
            generate_low_light: 是否自动生成低光图像
        """
        self.high_res_dir = high_res_dir
        self.low_light_dir = low_light_dir
        self.transform = transform
        self.auto_label = auto_label
        self.use_generate_low_light = generate_low_light
        
        # 初始化YOLO模型用于自动标注
        if auto_label:
            self.yolo_model = YOLO(model_path)
        
        # 初始化Zero-DCE用于生成低光图像
        if generate_low_light:
            # 这里使用简化版的Zero-DCE
            self.zero_dce = self._build_zero_dce()
        
        # 获取图像路径列表
        self.high_res_files = self._get_image_files(high_res_dir)
        
        if low_light_dir and os.path.exists(low_light_dir):
            self.low_light_files = self._get_image_files(low_light_dir)
            # 如果低光目录中有图像文件，尝试匹配
            if len(self.low_light_files) > 0:
                # 确保两个目录中的图像数量相同
                self.pair_files = self._match_files()
                # 如果没有匹配的图像对，使用高光图像生成
                if len(self.pair_files) == 0:
                    self.pair_files = [(f, None) for f in self.high_res_files]
                    print("未找到匹配的图像对，使用高光图像生成低光图像")
            else:
                # 如果低光目录中没有图像文件，使用高光图像生成
                self.pair_files = [(f, None) for f in self.high_res_files]
                print("低光目录中没有图像文件，使用高光图像生成低光图像")
        else:
            # 如果没有提供低光目录或目录不存在，使用高光图像生成
            self.pair_files = [(f, None) for f in self.high_res_files]
            print("低光目录不存在，使用高光图像生成低光图像")
        
        print(f"数据集初始化完成，共 {len(self.pair_files)} 对图像")
    
    def _get_image_files(self, directory):
        """
        获取目录中的图像文件
        
        Args:
            directory: 目录路径
            
        Returns:
            图像文件路径列表
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        files = []
        
        if not os.path.exists(directory):
            print(f"警告：目录 {directory} 不存在")
            return files
        
        try:
            for file in os.listdir(directory):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    files.append(os.path.join(directory, file))
            print(f"在目录 {directory} 中找到 {len(files)} 个图像文件")
        except Exception as e:
            print(f"读取目录 {directory} 时出错: {e}")
        
        return sorted(files)
    
    def _match_files(self):
        """
        匹配高清图和低光帧
        
        Returns:
            配对的文件路径列表
        """
        pair_files = []
        
        # 基于文件名匹配
        for high_res_file in self.high_res_files:
            filename = os.path.basename(high_res_file)
            low_light_file = os.path.join(self.low_light_dir, filename)
            
            if low_light_file in self.low_light_files:
                pair_files.append((high_res_file, low_light_file))
        
        if len(pair_files) == 0:
            print("警告：未找到匹配的图像对")
        
        return pair_files
    
    def _build_zero_dce(self):
        """
        构建简化版的Zero-DCE模型
        
        Returns:
            Zero-DCE模型
        """
        return ZeroDCE()
    
    def generate_low_light(self, high_res):
        """
        使用Zero-DCE生成低光图像
        
        Args:
            high_res: 高清图像
            
        Returns:
            生成的低光图像
        """
        if not self.use_generate_low_light:
            return high_res
        
        # 转换为张量
        img_tensor = torch.from_numpy(high_res).permute(2, 0, 1).unsqueeze(0).float()
        
        # 生成低光图像
        with torch.no_grad():
            low_light_tensor = self.zero_dce(img_tensor)
        
        # 转换回 numpy 数组
        low_light = low_light_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype('uint8')
        
        return low_light
    
    def _apply_augmentation(self, high_res, low_light):
        """
        应用数据增强
        
        Args:
            high_res: 高清图像
            low_light: 低光图像
            
        Returns:
            增强后的图像对
        """
        # 随机水平翻转
        if np.random.random() > 0.5:
            high_res = cv2.flip(high_res, 1)
            low_light = cv2.flip(low_light, 1)
        
        # 随机裁剪
        if np.random.random() > 0.5:
            h, w = high_res.shape[:2]
            crop_size = int(min(h, w) * 0.8)
            x = np.random.randint(0, w - crop_size)
            y = np.random.randint(0, h - crop_size)
            
            high_res = high_res[y:y+crop_size, x:x+crop_size]
            low_light = low_light[y:y+crop_size, x:x+crop_size]
        
        # 亮度扰动
        if np.random.random() > 0.5:
            brightness_factor = 0.8 + np.random.random() * 0.4  # 0.8-1.2
            high_res = np.clip(high_res * brightness_factor, 0, 255).astype(np.uint8)
            low_light = np.clip(low_light * brightness_factor, 0, 255).astype(np.uint8)
        
        # 高斯噪声
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 10, low_light.shape).astype(np.uint8)
            low_light = np.clip(low_light + noise, 0, 255).astype(np.uint8)
        
        # 调整大小
        high_res = cv2.resize(high_res, (Config.INPUT_SIZE, Config.INPUT_SIZE))
        low_light = cv2.resize(low_light, (Config.INPUT_SIZE, Config.INPUT_SIZE))
        
        return high_res, low_light
    
    def _auto_label(self, image):
        """
        使用YOLO模型自动标注图像
        
        Args:
            image: 输入图像
            
        Returns:
            标注列表，格式为 [[cls, x, y, w, h], ...]
        """
        if not self.auto_label:
            return []
        
        # 使用YOLO模型检测
        results = self.yolo_model(image, verbose=False)
        
        labels = []
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    # 获取边界框
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    # 获取类别
                    cls = int(box.cls[0].item())
                    # 计算中心坐标和宽高（归一化）
                    h, w = image.shape[:2]
                    center_x = (x1 + x2) / 2 / w
                    center_y = (y1 + y2) / 2 / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    
                    # 确保坐标在有效范围内
                    center_x = max(0, min(1, center_x))
                    center_y = max(0, min(1, center_y))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))
                    
                    labels.append([cls, center_x, center_y, width, height])
        
        return labels
    
    def __getitem__(self, idx):
        """
        获取数据项
        
        Args:
            idx: 索引
            
        Returns:
            高清图像、低光图像和标签
        """
        high_res_path, low_light_path = self.pair_files[idx]
        
        # 打印当前处理的图像
        print(f"处理图像 {idx+1}/{len(self.pair_files)}: {os.path.basename(high_res_path)}")
        
        # 加载图像
        try:
            high_res = cv2.imread(high_res_path)
            
            # 加载或生成低光图像
            if low_light_path:
                low_light = cv2.imread(low_light_path)
                if low_light is None:
                    # 如果低光图像加载失败，生成低光图像
                    print(f"生成低光图像: {os.path.basename(high_res_path)}")
                    low_light = self.generate_low_light(high_res)
            else:
                # 如果没有低光图像路径，生成低光图像
                print(f"生成低光图像: {os.path.basename(high_res_path)}")
                low_light = self.generate_low_light(high_res)
            
            if high_res is None:
                print(f"警告：无法加载高清图像 {high_res_path}")
                # 返回随机图像作为占位符
                high_res = np.zeros((Config.INPUT_SIZE, Config.INPUT_SIZE, 3), dtype=np.uint8)
                low_light = np.zeros((Config.INPUT_SIZE, Config.INPUT_SIZE, 3), dtype=np.uint8)
                labels = []
            else:
                # 应用数据增强
                high_res, low_light = self._apply_augmentation(high_res, low_light)
                
                # 自动标注
                labels = self._auto_label(high_res)
                print(f"标注完成，检测到 {len(labels)} 个目标")
            
            # 转换为张量
            high_res = torch.from_numpy(high_res).permute(2, 0, 1).float()
            low_light = torch.from_numpy(low_light).permute(2, 0, 1).float()
            
            # 确保标签格式一致
            # 对于DataLoader的collate_fn，我们需要确保每个样本的标签格式一致
            # 这里我们返回一个字典，包含标签列表
            return high_res, low_light, labels
        except Exception as e:
            print(f"加载图像时出错: {e}")
            # 返回随机图像作为占位符
            high_res = torch.zeros(3, Config.INPUT_SIZE, Config.INPUT_SIZE)
            low_light = torch.zeros(3, Config.INPUT_SIZE, Config.INPUT_SIZE)
            return high_res, low_light, []
    
    def __len__(self):
        """
        获取数据集长度
        
        Returns:
            数据集长度
        """
        return len(self.pair_files)
