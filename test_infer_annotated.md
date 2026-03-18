# test_infer.py 完整注释

```python
# 测试推理模块
# 对比原生 YOLO 和增强 YOLO 在低光照条件下的识别效果
# 原理：通过降低图像亮度模拟低光环境，对比两种模型的性能

import os  # 操作系统接口
import cv2  # OpenCV 计算机视觉库
import numpy as np  # 数值计算库
import torch  # PyTorch 框架
import torch.nn as nn  # 神经网络模块
from ultralytics import YOLO  # YOLO 模型
from yolo_transformer import YOLOTransformerLowLight  # 增强模型
from high_res_cache import HighResFeatureCache  # 特征缓存
from light_enhance import LightEnhance  # 低光增强
from config import Config  # 配置参数
import torch.nn.functional as F  # 函数式接口

class TestInfer:
    """
    测试推理类
    用于对比原生 YOLO 和增强 YOLO 在低光照条件下的检测效果
    
    测试流程:
    1. 加载高清图并提取特征
    2. 降低图像亮度模拟低光环境
    3. 使用原生 YOLO 检测
    4. 使用增强 YOLO 检测
    5. 对比结果并可视化
    """
    
    def __init__(self, model_path='yolo26n.pt', 
                 enhanced_model_path='/Users/liuqi/Documents/transformer_yolo/runs/train/test_train_1773657540/best.pt'):
        """
        初始化测试推理类
        
        Args:
            model_path: 预训练 YOLO 模型路径
            enhanced_model_path: 训练好的增强模型路径
        """
        # 设备选择 - 优先 GPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')  # NVIDIA GPU
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')  # Apple Silicon
        else:
            self.device = torch.device('cpu')  # CPU
        
        # 初始化原生 YOLO
        self.native_yolo = YOLO(model_path)
        
        # 初始化增强 YOLO
        self.enhanced_yolo = YOLOTransformerLowLight(model_path)
        self.enhanced_yolo.to(self.device)
        
        # 加载训练好的增强模型
        if os.path.exists(enhanced_model_path):
            try:
                # 加载权重
                state_dict = torch.load(
                    enhanced_model_path, 
                    map_location=self.device, 
                    weights_only=False
                )
                # 加载参数
                self.enhanced_yolo.load_state_dict(state_dict, strict=False)
                print(f"成功加载增强模型：{enhanced_model_path}")
            except Exception as e:
                print(f"加载增强模型失败：{e}")
        else:
            print(f"增强模型文件不存在：{enhanced_model_path}")
        
        # 初始化缓存
        self.cache = HighResFeatureCache()
        
        # 初始化增强模块
        self.light_enhance = LightEnhance()
        
        print(f"测试推理初始化完成，使用设备：{self.device}")
    
    def load_image(self, image_path):
        """
        加载图像
        
        Args:
            image_path: 图像路径
        
        Returns:
            加载的图像 (BGR 格式)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像：{image_path}")
        return image
    
    def preprocess_image(self, image):
        """
        预处理图像
        
        Args:
            image: 输入图像 (BGR 格式)
        
        Returns:
            预处理后的张量 (RGB 格式，归一化)
        """
        # BGR → RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 归一化到 [0, 1]
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        
        # 增加 batch 维度
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor
    
    def reduce_brightness(self, image, factor=0.3):
        """
        降低图像亮度 - 模拟低光环境
        
        Args:
            image: 输入图像 (BGR 格式)
            factor: 亮度降低因子 (0-1)
                   0.3 表示降低到原来的 30%
        
        Returns:
            低亮度图像 (BGR 格式)
        """
        # BGR → HSV
        # H: 色调，S: 饱和度，V: 亮度
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 降低 V 通道 (亮度)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        
        # HSV → BGR
        low_light_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return low_light_image
    
    def extract_features(self, image):
        """
        提取图像特征 - 用于高清缓存
        
        Args:
            image: 输入图像 (BGR 格式)
        
        Returns:
            特征向量 (numpy 数组)
        """
        # 预处理
        image_tensor = self.preprocess_image(image)
        
        # 提取特征
        with torch.no_grad():
            # 1. 增强
            enhanced = self.enhanced_yolo.light_enhance(image_tensor)
            
            # 2. Backbone 提取特征
            features = self.enhanced_yolo.backbone(enhanced)
            
            # 3. 全局池化
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1)).squeeze()
            
            # 4. 维度调整
            if features.shape[-1] != Config.FEATURE_DIM:
                feat_adjust = nn.Linear(features.shape[-1], Config.FEATURE_DIM).to(self.device)
                features = feat_adjust(features)
        
        # 转换为 numpy 并移到 CPU
        return features.cpu().numpy()
    
    def detect_with_native_yolo(self, image):
        """
        使用原生 YOLO 进行检测
        
        Args:
            image: 输入图像 (BGR 格式)
        
        Returns:
            检测结果
        """
        results = self.native_yolo(image)
        return results
    
    def detect_with_enhanced_yolo(self, image, high_res_feat):
        """
        使用增强 YOLO 进行检测 (完整流程)
        
        Args:
            image: 输入图像 (BGR 格式)
            high_res_feat: 高清特征 (numpy 数组)
        
        Returns:
            检测结果
        """
        try:
            # 转换为 tensor
            high_res_feat_tensor = torch.from_numpy(high_res_feat).unsqueeze(0).to(self.device)
            
            # 调用增强模型的 detect 方法
            print("使用增强模型的 detect 方法")
            results = self.enhanced_yolo.detect(image, high_res_feat_tensor)
            print("使用增强模型完成检测")
            
            print(f"增强 YOLO 检测结果：{results}")
            return results
        except Exception as e:
            # fallback 到原生 YOLO
            print(f"增强 YOLO 检测失败：{e}")
            import traceback
            traceback.print_exc()
            results = self.native_yolo(image)
            return results
    
    def draw_detections(self, image, results, color=(0, 255, 0)):
        """
        绘制检测结果
        
        Args:
            image: 输入图像
            results: 检测结果
            color: 绘制颜色 (BGR 格式)
                   默认绿色 (0,255,0)
        
        Returns:
            绘制了检测结果的图像
        """
        # 复制图像
        output_image = image.copy()
        
        # 绘制检测框
        if isinstance(results, list) and len(results) > 0:
            for result in results:
                if hasattr(result, 'boxes'):
                    # 原生 YOLO 格式
                    boxes = result.boxes
                    for box in boxes:
                        # 获取坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # 获取置信度
                        conf = box.conf[0].cpu().numpy()
                        
                        # 获取类别
                        cls = box.cls[0].cpu().numpy()
                        
                        # 绘制矩形框
                        cv2.rectangle(output_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # 绘制标签
                        label = f"{self.native_yolo.names[int(cls)]}: {conf:.2f}"
                        cv2.putText(output_image, label, (int(x1), int(y1) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                elif isinstance(result, torch.Tensor) or isinstance(result, np.ndarray):
                    # Tensor/ numpy 格式
                    if len(result) > 0:
                        for detection in result:
                            if len(detection) >= 6:
                                # [x1, y1, x2, y2, conf, cls]
                                x1, y1, x2, y2, conf, cls = detection
                                cv2.rectangle(output_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                                label = f"{self.native_yolo.names[int(cls)]}: {conf:.2f}"
                                cv2.putText(output_image, label, (int(x1), int(y1) - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output_image
    
    def compare_results(self, native_results, enhanced_results):
        """
        比较两种方法的检测结果
        
        Args:
            native_results: 原生 YOLO 结果
            enhanced_results: 增强 YOLO 结果
        
        Returns:
            比较结果字典
        """
        # 统计原生 YOLO 的检测数量
        native_count = 0
        if isinstance(native_results, list):
            for result in native_results:
                if hasattr(result, 'boxes'):
                    native_count += len(result.boxes)
        
        # 统计增强 YOLO 的检测数量
        enhanced_count = 0
        if isinstance(enhanced_results, list):
            for result in enhanced_results:
                if hasattr(result, 'boxes'):
                    enhanced_count += len(result.boxes)
                elif isinstance(result, (torch.Tensor, np.ndarray)):
                    if len(result) > 0:
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
        
        Returns:
            比较结果
        """
        print(f"\n测试图像：{image_path}")
        
        # 1. 加载图像
        original_image = self.load_image(image_path)
        
        # 2. 提取高清特征
        print("提取高清特征...")
        high_res_feat = self.extract_features(original_image)
        print(f"高清特征提取完成，特征维度：{high_res_feat.shape}")
        
        # 3. 降低亮度
        print("降低图像亮度...")
        low_light_image = self.reduce_brightness(original_image, brightness_factor)
        
        # 4. 原生 YOLO 检测
        print("使用原生 YOLO 检测低光图像...")
        import time
        native_start = time.time()
        native_results = self.detect_with_native_yolo(low_light_image)
        native_time = time.time() - native_start
        
        # 5. 增强 YOLO 检测
        print("使用增强 YOLO 检测低光图像...")
        enhanced_start = time.time()
        enhanced_results = self.detect_with_enhanced_yolo(low_light_image, high_res_feat)
        enhanced_time = time.time() - enhanced_start
        
        # 6. 绘制结果
        print("绘制检测结果...")
        native_drawn = self.draw_detections(low_light_image, native_results, color=(0, 255, 0))  # 绿色
        enhanced_drawn = self.draw_detections(low_light_image, enhanced_results, color=(0, 0, 255))  # 红色
        
        # 7. 比较结果
        comparison = self.compare_results(native_results, enhanced_results)
        print("\n检测结果对比:")
        print(f"原生 YOLO 检测目标数：{comparison['native_count']}")
        print(f"原生 YOLO 用时：{native_time*1000:.1f}ms")
        print(f"增强 YOLO 检测目标数：{comparison['enhanced_count']}")
        print(f"增强 YOLO 用时：{enhanced_time*1000:.1f}ms")
        print(f"提升百分比：{comparison['improvement']:.2f}%")
        
        # 8. 保存结果
        output_dir = 'test_results'
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存原始图像
        original_path = os.path.join(output_dir, 'original.jpg')
        cv2.imwrite(original_path, original_image)
        print(f"保存原始图像到：{original_path}")
        
        # 保存低光图像
        low_light_path = os.path.join(output_dir, 'low_light.jpg')
        cv2.imwrite(low_light_path, low_light_image)
        print(f"保存低光图像到：{low_light_path}")
        
        # 保存原生 YOLO 结果
        native_path = os.path.join(output_dir, 'native_yolo_result.jpg')
        cv2.imwrite(native_path, native_drawn)
        print(f"保存原生 YOLO 检测结果到：{native_path}")
        
        # 保存增强 YOLO 结果
        enhanced_path = os.path.join(output_dir, 'enhanced_yolo_result.jpg')
        cv2.imwrite(enhanced_path, enhanced_drawn)
        print(f"保存增强 YOLO 检测结果到：{enhanced_path}")
        
        return comparison

def test_infer(image_path):
    """
    测试推理入口函数
    
    Args:
        image_path: 图像路径
    
    Returns:
        比较结果
    """
    try:
        tester = TestInfer()
        result = tester.run_test(image_path)
        return result
    except Exception as e:
        print(f"测试失败：{e}")
        raise

if __name__ == "__main__":
    # 参数解析
    import argparse
    
    parser = argparse.ArgumentParser(description="测试低光照目标检测增强")
    parser.add_argument('--image', type=str, 
                       default='sample_data/high_res/000000001237.jpg', 
                       help='测试图像路径')
    parser.add_argument('--brightness', type=float, 
                       default=0.3, 
                       help='亮度降低因子')
    
    args = parser.parse_args()
    
    # 运行测试
    test_infer(args.image)
```
