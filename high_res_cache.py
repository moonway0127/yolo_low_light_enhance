# 高清特征缓存模块
# 离线提取和缓存高清图特征

import os
import pickle
import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from config import Config

class HighResFeatureCache:
    """
    高清特征缓存类
    用于离线提取和缓存高清图特征，供实时推理使用
    """
    
    def __init__(self):
        """
        初始化高清特征缓存
        """
        # 加载 YOLO26n 模型
        self.model = YOLO('yolo26n.pt')
        # 特征压缩层
        self.feature_compress = torch.nn.Conv2d(512, Config.FEATURE_DIM, kernel_size=1)
        # 缓存特征
        self.cache = {}
        # 加载本地缓存
        self.load_cache()
    
    def extract_high_res_feat(self, image_path):
        """
        离线提取高清图特征
        
        Args:
            image_path: 高清图像路径
            
        Returns:
            压缩后的特征向量
        """
        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法加载图像: {image_path}")
        
        # 使用 YOLO 提取特征
        with torch.no_grad():
            # 获取模型的 backbone 输出
            results = self.model(image, verbose=False)
            # 提取特征图
            feat = results[0].orig_img
            # 转换为张量
            feat = torch.from_numpy(feat).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            # 通过压缩层
            compressed_feat = self.feature_compress(feat)
            # 全局池化
            global_feat = torch.nn.functional.adaptive_avg_pool2d(compressed_feat, (1, 1)).squeeze()
        
        return global_feat.numpy()
    
    def load_cache(self):
        """
        加载内存 / 本地缓存特征
        """
        try:
            if os.path.exists(Config.CACHE_PATH):
                with open(Config.CACHE_PATH, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"成功加载缓存: {Config.CACHE_PATH}")
            else:
                print(f"缓存文件不存在: {Config.CACHE_PATH}")
        except Exception as e:
            print(f"加载缓存失败: {e}")
            self.cache = {}
    
    def save_cache(self):
        """
        保存特征至内存 / 本地文件（pickle 格式）
        """
        try:
            with open(Config.CACHE_PATH, 'wb') as f:
                pickle.dump(self.cache, f)
            print(f"成功保存缓存: {Config.CACHE_PATH}")
        except Exception as e:
            print(f"保存缓存失败: {e}")
    
    def is_cache_valid(self, current_image, cache_key):
        """
        基于 SIFT 特征匹配 + 时间戳校验缓存有效性
        
        Args:
            current_image: 当前低光帧
            cache_key: 缓存键
            
        Returns:
            bool: 缓存是否有效
        """
        # 检查缓存是否存在
        if cache_key not in self.cache:
            return False
        
        # 检查时间戳
        timestamp = self.cache[cache_key]['timestamp']
        if time.time() - timestamp > Config.CACHE_TIMEOUT:
            return False
        
        # 检查 SIFT 特征匹配
        try:
            # 加载缓存的高清图
            high_res_path = self.cache[cache_key]['path']
            if not os.path.exists(high_res_path):
                return False
            
            high_res_img = cv2.imread(high_res_path, cv2.IMREAD_GRAYSCALE)
            current_img = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
            
            # 初始化 SIFT
            sift = cv2.SIFT_create()
            
            # 检测特征点和描述符
            kp1, des1 = sift.detectAndCompute(high_res_img, None)
            kp2, des2 = sift.detectAndCompute(current_img, None)
            
            # 特征匹配
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            
            # 应用 ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            
            # 计算匹配率
            match_ratio = len(good_matches) / max(len(kp1), len(kp2)) if max(len(kp1), len(kp2)) > 0 else 0
            
            return match_ratio >= Config.SCENE_MATCH_THRESHOLD
        except Exception as e:
            print(f"SIFT 特征匹配失败: {e}")
            return False
    
    def update_cache(self, image_path, cache_key):
        """
        缓存失效时重新提取并更新
        
        Args:
            image_path: 高清图像路径
            cache_key: 缓存键
            
        Returns:
            新提取的特征
        """
        try:
            # 提取特征
            feat = self.extract_high_res_feat(image_path)
            # 更新缓存
            self.cache[cache_key] = {
                'feature': feat,
                'path': image_path,
                'timestamp': time.time()
            }
            # 保存缓存
            self.save_cache()
            print(f"更新缓存: {cache_key}")
            return feat
        except Exception as e:
            print(f"更新缓存失败: {e}")
            return None
    
    def detect_motion(self, prev_frame, current_frame, threshold=300):
        """
        检测是否有移动的物体
        
        Args:
            prev_frame: 之前的帧
            current_frame: 当前帧
            threshold: 移动阈值
            
        Returns:
            bool: 是否检测到移动
        """
        # 转换为灰度图
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # 计算帧差
        frame_diff = cv2.absdiff(prev_gray, current_gray)
        
        # 二值化
        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        
        # 计算非零像素数
        non_zero_count = cv2.countNonZero(thresh)
        
        return non_zero_count > threshold
    
    def detect_person(self, frame):
        """
        检测帧中是否有人
        
        Args:
            frame: 输入帧
            
        Returns:
            bool: 是否检测到人
        """
        # 使用YOLO模型检测人
        results = self.model(frame, verbose=False)
        
        # 检查是否检测到人（COCO数据集中person的类别ID是0）
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    cls = int(box.cls[0].item())
                    if cls == 0:  # person
                        return True
        
        return False
    
    def calculate_clarity(self, image):
        """
        计算图像清晰度
        
        Args:
            image: 输入图像
            
        Returns:
            float: 清晰度分数
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用拉普拉斯算子计算清晰度
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        clarity = laplacian.var()
        
        return clarity
    
    def should_update_cache(self, current_frame, cache_key):
        """
        判断是否应该更新缓存
        
        Args:
            current_frame: 当前帧
            cache_key: 缓存键
            
        Returns:
            bool: 是否应该更新
        """
        # 检查缓存是否存在
        if cache_key not in self.cache:
            return True
        
        # 检查是否有移动的物体
        # 这里使用当前帧和缓存的高清图进行比较
        high_res_path = self.cache[cache_key]['path']
        if not os.path.exists(high_res_path):
            return True
        
        high_res_img = cv2.imread(high_res_path)
        if high_res_img is None:
            return True
        
        # 检测移动
        has_motion = self.detect_motion(high_res_img, current_frame)
        if has_motion:
            return False  # 有移动，不更新
        
        # 检测是否有人
        has_person = self.detect_person(current_frame)
        if has_person:
            return False  # 有人，不更新
        
        # 计算清晰度
        current_clarity = self.calculate_clarity(current_frame)
        cache_clarity = self.calculate_clarity(high_res_img)
        
        # 检查清晰度是否相当（差值在20%以内）
        clarity_ratio = min(current_clarity, cache_clarity) / max(current_clarity, cache_clarity)
        return clarity_ratio >= 0.8
    
    def timed_update_cache(self, current_frame, cache_key, update_interval=3600):
        """
        定时更新缓存
        
        Args:
            current_frame: 当前帧
            cache_key: 缓存键
            update_interval: 更新间隔（秒）
            
        Returns:
            bool: 是否更新成功
        """
        # 检查是否到了更新时间
        if cache_key in self.cache:
            timestamp = self.cache[cache_key]['timestamp']
            if time.time() - timestamp < update_interval:
                return False
        
        # 检查是否应该更新
        if not self.should_update_cache(current_frame, cache_key):
            return False
        
        # 保存当前帧作为新的高清图
        temp_high_res_path = os.path.join(os.path.dirname(Config.CACHE_PATH), f"temp_high_res_{cache_key}.jpg")
        cv2.imwrite(temp_high_res_path, current_frame)
        
        # 更新缓存
        try:
            self.update_cache(temp_high_res_path, cache_key)
            print(f"定时更新缓存: {cache_key}")
            return True
        except Exception as e:
            print(f"定时更新缓存失败: {e}")
            return False
    
    def get_feature(self, current_image, cache_key, high_res_path=None):
        """
        获取特征，如果缓存无效则更新
        
        Args:
            current_image: 当前低光帧
            cache_key: 缓存键
            high_res_path: 高清图像路径（当需要更新缓存时使用）
            
        Returns:
            特征向量
        """
        # 定时更新缓存
        self.timed_update_cache(current_image, cache_key)
        
        # 检查缓存有效性
        if self.is_cache_valid(current_image, cache_key):
            return self.cache[cache_key]['feature']
        else:
            # 如果提供了高清图像路径，则更新缓存
            if high_res_path and os.path.exists(high_res_path):
                return self.update_cache(high_res_path, cache_key)
            else:
                return None
