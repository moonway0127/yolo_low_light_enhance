# 实时推理模块
# 支持视频流、摄像头和图片输入的低光照目标检测

import os
import time
import logging
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from yolo_transformer import YOLOTransformerLowLight
from high_res_cache import HighResFeatureCache
from config import Config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(Config.LOG_DIR, 'infer.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Inferencer:
    """
    推理器类
    用于实时低光照视频目标检测
    """
    
    def __init__(self, model_path='yolo26n.pt', cache_path=None):
        """
        初始化推理器
        
        Args:
            model_path: 模型路径
            cache_path: 缓存文件路径
        """
        # 初始化模型
        self.model = YOLOTransformerLowLight(model_path)
        # 优先使用MPS（Apple Silicon GPU），然后是CUDA，最后是CPU
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)
        # 设置模型为评估模式
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 初始化原生 YOLO 模型（用于降级）
        self.native_yolo = YOLO(model_path)
        
        # 初始化高清特征缓存
        self.cache = HighResFeatureCache()
        if cache_path and os.path.exists(cache_path):
            self.cache.CACHE_PATH = cache_path
            self.cache.load_cache()
        
        # 性能统计
        self.fps_list = []
        self.detection_count = 0
        
        logger.info(f"推理器初始化完成，使用设备: {self.device}")
    
    def preprocess(self, frame):
        """
        预处理图像
        
        Args:
            frame: 输入图像
            
        Returns:
            预处理后的图像
        """
        # 调整大小
        frame = cv2.resize(frame, (Config.INPUT_SIZE, Config.INPUT_SIZE))
        # 转换为张量
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        return frame_tensor
    
    def infer_image(self, image, high_res_feat=None):
        """
        推理单张图像
        
        Args:
            image: 输入图像
            high_res_feat: 高清缓存特征
            
        Returns:
            检测结果
        """
        start_time = time.time()
        
        # 预处理
        image_tensor = self.preprocess(image)
        
        # 推理
        with torch.no_grad():
            if high_res_feat is not None:
                # 使用融合增强
                print("使用特征增强模块")
                # 低光增强
                enhanced_frame = self.model.light_enhance(image_tensor)
                # 提取低光特征
                low_light_feat = self.model.backbone(enhanced_frame)
                # 生成暗部掩码
                dark_mask = self.model.get_dark_mask(image_tensor)
                import torch.nn.functional as F
                dark_mask = F.interpolate(dark_mask, size=(low_light_feat.shape[2], low_light_feat.shape[3]), mode='bilinear', align_corners=False)
                # 特征融合
                fused_feat = self.model.fusion(low_light_feat, high_res_feat, dark_mask)
                # 通过检测头
                outputs = self.model.head(fused_feat)
                
                # 将增强后的图像转换回numpy格式用于检测
                enhanced_image_np = enhanced_frame.squeeze().permute(1, 2, 0).cpu().numpy()
                enhanced_image_np = np.clip(enhanced_image_np, 0, 1)
                enhanced_image_np = (enhanced_image_np * 255).astype(np.uint8)
                enhanced_image_bgr = cv2.cvtColor(enhanced_image_np, cv2.COLOR_RGB2BGR)
                
                # 使用原生YOLO对增强后的图像进行检测
                results = self.native_yolo(enhanced_image_bgr)
            else:
                # 降级为原生 YOLO
                results = self.native_yolo(image)
        
        # 计算推理时间
        infer_time = time.time() - start_time
        self.fps_list.append(1 / infer_time)
        if len(self.fps_list) > 30:
            self.fps_list.pop(0)
        
        return results
    
    def visualize(self, image, results, use_enhance=True):
        """
        可视化检测结果
        
        Args:
            image: 原始图像
            results: 检测结果
            use_enhance: 是否使用增强
            
        Returns:
            可视化后的图像
        """
        # 复制图像
        vis_image = image.copy()
        
        # 绘制检测框
        if isinstance(results, list) and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    # 获取置信度
                    conf = box.conf[0].item()
                    # 获取类别
                    cls = int(box.cls[0].item())
                    # 获取类别名称
                    cls_name = result.names[cls]
                    
                    # 绘制边界框
                    color = (0, 255, 0) if use_enhance else (0, 0, 255)
                    cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # 绘制标签
                    label = f"{cls_name}: {conf:.2f}"
                    cv2.putText(vis_image, label, (int(x1), int(y1) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 绘制 FPS
        fps = np.mean(self.fps_list) if self.fps_list else 0
        cv2.putText(vis_image, f"FPS: {fps:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 绘制增强标识
        mode = "Enhanced" if use_enhance else "Native"
        cv2.putText(vis_image, f"Mode: {mode}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return vis_image
    
    def process_video(self, video_path, output_path=None):
        """
        处理视频
        
        Args:
            video_path: 视频路径或摄像头ID
            output_path: 输出视频路径
        """
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频: {video_path}")
            return
        
        # 获取视频参数
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 创建输出视频
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 处理视频帧
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 获取高清特征
            high_res_feat = self.cache.get_feature(frame, 'default')
            use_enhance = high_res_feat is not None
            
            # 推理
            results = self.infer_image(frame, high_res_feat)
            
            # 可视化
            vis_frame = self.visualize(frame, results, use_enhance)
            
            # 显示
            cv2.imshow('Low Light Detection', vis_frame)
            
            # 保存输出
            if out:
                out.write(vis_frame)
            
            # 统计检测数量
            if isinstance(results, list) and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    self.detection_count += len(result.boxes)
            
            frame_count += 1
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 释放资源
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # 打印统计信息
        avg_fps = np.mean(self.fps_list) if self.fps_list else 0
        logger.info(f"视频处理完成，共处理 {frame_count} 帧")
        logger.info(f"平均 FPS: {avg_fps:.2f}")
        logger.info(f"总检测目标数: {self.detection_count}")
    
    def process_image(self, image_path, output_path=None):
        """
        处理单张图像
        
        Args:
            image_path: 图像路径
            output_path: 输出图像路径
        """
        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法加载图像: {image_path}")
            return
        
        # 获取高清特征
        high_res_feat = self.cache.get_feature(image, 'default')
        use_enhance = high_res_feat is not None
        
        # 推理
        results = self.infer_image(image, high_res_feat)
        
        # 可视化
        vis_image = self.visualize(image, results, use_enhance)
        
        # 显示
        cv2.imshow('Low Light Detection', vis_image)
        
        # 保存输出
        if output_path:
            cv2.imwrite(output_path, vis_image)
            logger.info(f"结果保存到: {output_path}")
        
        # 等待按键
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 打印统计信息
        if isinstance(results, list) and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                detection_count = len(result.boxes)
                logger.info(f"检测到 {detection_count} 个目标")

def infer_video(video_path, model_path='yolo26n.pt', cache_path=None, output_path=None):
    """
    视频推理入口函数
    
    Args:
        video_path: 视频路径或摄像头ID
        model_path: 模型路径
        cache_path: 缓存文件路径
        output_path: 输出视频路径
    """
    try:
        inferencer = Inferencer(model_path, cache_path)
        inferencer.process_video(video_path, output_path)
    except Exception as e:
        logger.error(f"推理失败: {e}")
        raise

def infer_image(image_path, model_path='yolo26n.pt', cache_path=None, output_path=None):
    """
    图像推理入口函数
    
    Args:
        image_path: 图像路径
        model_path: 模型路径
        cache_path: 缓存文件路径
        output_path: 输出图像路径
    """
    try:
        inferencer = Inferencer(model_path, cache_path)
        inferencer.process_image(image_path, output_path)
    except Exception as e:
        logger.error(f"推理失败: {e}")
        raise
