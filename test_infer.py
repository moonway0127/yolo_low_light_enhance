# 测试推理模块
# 对比原生YOLO和增强模块后的YOLO在低光照条件下的识别效果

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from yolo_transformer import YOLOTransformerLowLight
from high_res_cache import HighResFeatureCache
from light_enhance import LightEnhance
from config import Config
import torch.nn.functional as F
class TestInfer:
    """
    测试推理类
    用于对比原生YOLO和增强模块后的YOLO在低光照条件下的识别效果
    """
    
    def __init__(self, model_path='yolo26n.pt', enhanced_model_path='/Users/liuqi/Documents/transformer_yolo/runs/train/test_train_1773657540/best.pt'):
        """
        初始化测试推理类
        
        Args:
            model_path: 预训练YOLO模型路径
            enhanced_model_path: 训练好的增强模型路径
        """
        # 设备选择 - 优先使用GPU或NPU，完全不考虑CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            # 没有GPU或NPU时才使用CPU
            self.device = torch.device('cpu')
        
        # 初始化原生YOLO模型
        self.native_yolo = YOLO(model_path)
        
        # 初始化增强后的YOLO模型
        self.enhanced_yolo = YOLOTransformerLowLight(model_path)
        self.enhanced_yolo.to(self.device)
        
        # 加载训练好的增强模型
        if os.path.exists(enhanced_model_path):
            try:
                # 加载模型权重
                state_dict = torch.load(enhanced_model_path, map_location=self.device, weights_only=False)
                # 直接加载所有参数
                self.enhanced_yolo.load_state_dict(state_dict, strict=False)
                print(f"成功加载增强模型: {enhanced_model_path}")
            except Exception as e:
                print(f"加载增强模型失败: {e}")
        else:
            print(f"增强模型文件不存在: {enhanced_model_path}")
        
        # 初始化高清特征缓存
        self.cache = HighResFeatureCache()
        
        # 初始化低光增强模块
        self.light_enhance = LightEnhance()
        
        print(f"测试推理初始化完成，使用设备: {self.device}")
    
    def load_image(self, image_path):
        """
        加载图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            加载的图像 (BGR格式)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        return image
    
    def preprocess_image(self, image):
        """
        预处理图像
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            预处理后的图像 (RGB格式，归一化)
        """
        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 归一化
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        # 添加批次维度
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        return image_tensor
    
    def reduce_brightness(self, image, factor=0.3):
        """
        降低图像亮度
        
        Args:
            image: 输入图像 (BGR格式)
            factor: 亮度降低因子 (0-1)
            
        Returns:
            低亮度图像 (BGR格式)
        """
        # 转换为HSV色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 降低V通道（亮度）
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        # 转换回BGR
        low_light_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return low_light_image
    
    def extract_features(self, image):
        """
        提取图像特征
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            提取的特征
        """
        # 预处理图像
        image_tensor = self.preprocess_image(image)
        
        # 使用增强模型的backbone提取特征
        with torch.no_grad():
            enhanced = self.enhanced_yolo.light_enhance(image_tensor)
            features = self.enhanced_yolo.backbone(enhanced)
            # 全局池化
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1)).squeeze()
            # 确保特征维度与配置匹配
            if features.shape[-1] != Config.FEATURE_DIM:
                # 使用线性层调整维度
                feat_adjust = nn.Linear(features.shape[-1], Config.FEATURE_DIM).to(self.device)
                features = feat_adjust(features)
        
        return features.cpu().numpy()
    
    def detect_with_native_yolo(self, image):
        """
        使用原生YOLO进行检测
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            检测结果
        """
        results = self.native_yolo(image)
        return results
    
    def detect_with_enhanced_yolo(self, image, high_res_feat):
        """
        使用增强后的YOLO进行检测（完整流程）
        
        Args:
            image: 输入图像 (BGR格式)
            high_res_feat: 高清特征
            
        Returns:
            检测结果
        """
        # 完整流程：直接调用detect方法，包含低光增强 + 特征融合 + 端到端推理
        try:
            # 转换高清特征为tensor
            high_res_feat_tensor = torch.from_numpy(high_res_feat).unsqueeze(0).to(self.device)
            
            # 直接调用增强模型的detect方法
            print("使用增强模型的detect方法")
            results = self.enhanced_yolo.detect(image, high_res_feat_tensor)
            print("使用增强模型完成检测")
            
            print(f"增强YOLO检测结果: {results}")
            print("使用完整流程增强YOLO检测完成")
            return results
        except Exception as e:
            print(f"增强YOLO检测失败: {e}")
            import traceback
            traceback.print_exc()
            # fallback到使用原始图像的原生YOLO检测
            results = self.native_yolo(image)
            return results
    
    def draw_detections(self, image, results, color=(0, 255, 0)):
        """
        在图像上绘制检测结果
        
        Args:
            image: 输入图像 (BGR格式)
            results: 检测结果
            color: 绘制颜色
            
        Returns:
            绘制了检测结果的图像
        """
        # 复制图像
        output_image = image.copy()
        
        # 绘制检测框
        if isinstance(results, list) and len(results) > 0:
            for result in results:
                if hasattr(result, 'boxes'):
                    # 处理原生YOLO的结果格式
                    boxes = result.boxes
                    for box in boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        # 获取置信度
                        conf = box.conf[0].cpu().numpy()
                        # 获取类别
                        cls = box.cls[0].cpu().numpy()
                        # 绘制边界框
                        cv2.rectangle(output_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        # 绘制标签
                        label = f"{self.native_yolo.names[int(cls)]}: {conf:.2f}"
                        cv2.putText(output_image, label, (int(x1), int(y1) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                elif isinstance(result, torch.Tensor) or isinstance(result, np.ndarray):
                    # 处理模型后处理的结果格式
                    if len(result) > 0:
                        for detection in result:
                            # 假设detection格式为 [x1, y1, x2, y2, conf, cls]
                            if len(detection) >= 6:
                                x1, y1, x2, y2, conf, cls = detection
                                # 绘制边界框
                                cv2.rectangle(output_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                                # 绘制标签
                                label = f"{self.native_yolo.names[int(cls)]}: {conf:.2f}"
                                cv2.putText(output_image, label, (int(x1), int(y1) - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output_image
    
    def compare_results(self, native_results, enhanced_results):
        """
        比较两种方法的检测结果
        
        Args:
            native_results: 原生YOLO的检测结果
            enhanced_results: 增强YOLO的检测结果
            
        Returns:
            比较结果字典
        """
        # 统计原生YOLO的检测数量
        native_count = 0
        if isinstance(native_results, list):
            for result in native_results:
                if hasattr(result, 'boxes'):
                    native_count += len(result.boxes)
        
        # 统计增强YOLO的检测数量
        enhanced_count = 0
        if isinstance(enhanced_results, list):
            for result in enhanced_results:
                if hasattr(result, 'boxes'):
                    enhanced_count += len(result.boxes)
                elif isinstance(result, torch.Tensor) or isinstance(result, np.ndarray):
                    if len(result) > 0:
                        enhanced_count += len(result)
                elif len(result) > 0:
                    enhanced_count += len(result)
        
        # 计算提升百分比
        if native_count > 0:
            improvement = ((enhanced_count - native_count) / native_count) * 100
        else:
            improvement = 100 if enhanced_count > 0 else 0
        
        return {
            'native_count': native_count,
            'enhanced_count': enhanced_count,
            'improvement': improvement
        }
    
    def run_test(self, image_path, brightness_factor=0.3):
        """
        运行完整测试
        
        Args:
            image_path: 图像路径
            brightness_factor: 亮度降低因子
        """
        print(f"\n测试图像：{image_path}")
        
        # 加载图像
        original_image = self.load_image(image_path)
        
        # 提取高清特征并缓存
        print("提取高清特征...")
        high_res_feat = self.extract_features(original_image)
        print(f"高清特征提取完成，特征维度：{high_res_feat.shape}")
        
        # 降低图像亮度
        print("降低图像亮度...")
        low_light_image = self.reduce_brightness(original_image, brightness_factor)
        
        # 使用原生 YOLO 检测低光图像
        print("使用原生 YOLO 检测低光图像...")
        import time
        native_start = time.time()
        native_results = self.detect_with_native_yolo(low_light_image)
        native_time = time.time() - native_start
        
        # 使用增强 YOLO 检测低光图像
        print("使用增强 YOLO 检测低光图像...")
        enhanced_start = time.time()
        enhanced_results = self.detect_with_enhanced_yolo(low_light_image, high_res_feat)
        enhanced_time = time.time() - enhanced_start
        
        # 绘制检测结果
        print("绘制检测结果...")
        native_drawn = self.draw_detections(low_light_image, native_results, color=(0, 255, 0))  # 绿色
        enhanced_drawn = self.draw_detections(low_light_image, enhanced_results, color=(0, 0, 255))  # 红色
        
        # 比较结果
        comparison = self.compare_results(native_results, enhanced_results)
        print("\n检测结果对比:")
        print(f"原生 YOLO 检测目标数：{comparison['native_count']}")
        print(f"原生 YOLO 用时：{native_time*1000:.1f}ms")
        print(f"增强 YOLO 检测目标数：{comparison['enhanced_count']}")
        print(f"增强 YOLO 用时：{enhanced_time*1000:.1f}ms")
        print(f"提升百分比：{comparison['improvement']:.2f}%")
        
        # 保存结果图像
        output_dir = 'test_results'
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存原始图像
        original_path = os.path.join(output_dir, 'original.jpg')
        cv2.imwrite(original_path, original_image)
        print(f"保存原始图像到: {original_path}")
        
        # 保存低光图像
        low_light_path = os.path.join(output_dir, 'low_light.jpg')
        cv2.imwrite(low_light_path, low_light_image)
        print(f"保存低光图像到: {low_light_path}")
        
        # 保存原生YOLO检测结果
        native_path = os.path.join(output_dir, 'native_yolo_result.jpg')
        cv2.imwrite(native_path, native_drawn)
        print(f"保存原生YOLO检测结果到: {native_path}")
        
        # 保存增强YOLO检测结果
        enhanced_path = os.path.join(output_dir, 'enhanced_yolo_result.jpg')
        cv2.imwrite(enhanced_path, enhanced_drawn)
        print(f"保存增强YOLO检测结果到: {enhanced_path}")
        
        return comparison

def test_infer(image_path):
    """
    测试推理的入口函数
    
    Args:
        image_path: 图像路径
    """
    try:
        tester = TestInfer()
        result = tester.run_test(image_path)
        return result
    except Exception as e:
        print(f"测试失败: {e}")
        raise

if __name__ == "__main__":
    # 示例调用
    import argparse
    
    parser = argparse.ArgumentParser(description="测试低光照目标检测增强")
    parser.add_argument('--image', type=str, default='sample_data/high_res/000000001237.jpg', help='测试图像路径')
    parser.add_argument('--brightness', type=float, default=0.3, help='亮度降低因子')
    
    args = parser.parse_args()
    
    test_infer(args.image)
