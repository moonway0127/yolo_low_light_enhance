# yolo_transformer.py 完整注释

```python
# 核心检测模型
# 整合 YOLO26n Backbone + LightEnhance + MobileViTAttention + 自定义检测头
# 原理：将低光增强、特征融合和目标检测整合到统一的端到端框架中

import torch  # PyTorch 深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 函数式接口
import numpy as np  # 数值计算库
import cv2  # OpenCV 计算机视觉库
from ultralytics import YOLO  # YOLO 目标检测模型
from light_enhance import LightEnhance  # 低光增强模块
from fusion_module import MobileViTAttention  # Transformer 融合模块
from config import Config  # 配置参数

class CustomDetectHead(nn.Module):
    """
    自定义检测头
    用于处理融合特征并输出标准格式的检测结果
    
    核心原理:
    - 基于 YOLO 的检测头设计
    - 输出边界框、置信度和类别概率
    - 支持多尺度检测 (3 个 anchor)
    """
    
    def __init__(self, num_classes=80):
        """
        初始化检测头
        
        Args:
            num_classes: 类别数量，默认 80(COCO 数据集)
        """
        # 调用父类构造函数
        super(CustomDetectHead, self).__init__()
        
        # 保存类别数量
        self.num_classes = num_classes
        
        # 检测层：1x1 卷积
        # 输入：16 通道 (融合特征)
        # 输出：(5 + num_classes) * 3 通道
        #   - 5: 边界框 4 个参数 (x,y,w,h) + 1 个置信度
        #   - num_classes: 类别概率
        #   - 3: 3 个 anchor(每个网格点预测 3 个边界框)
        self.detect = nn.Conv2d(16, (5 + num_classes) * 3, kernel_size=1)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 融合特征 (B, 16, H, W)
        
        Returns:
            检测结果 (B, num_anchors, grid_h, grid_w, 5+num_classes)
        """
        # 确保特征内存连续
        # 原因：某些操作 (如 view) 需要连续的内存布局
        x = x.contiguous()
        
        # 通过检测层
        # 输出形状：(B, (5+num_classes)*3, H, W)
        out = self.detect(x)
        
        # 获取批次大小
        batch_size = out.shape[0]
        
        # 获取类别数量
        num_classes = self.num_classes
        
        # 调整输出格式
        # 目标格式：(B, num_anchors, 5+num_classes, grid_h, grid_w)
        # view: 重新调整形状
        #   out.shape[-1] 是 W 维度
        #   -1 自动计算 H 维度
        out = out.view(batch_size, 3, 5 + num_classes, -1, out.shape[-1])
        
        # permute: 调整维度顺序
        #   从 (B, 3, 5+num_classes, H, W) 到 (B, 3, H, W, 5+num_classes)
        # 原因：方便后续处理，将通道维度移到最后
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        
        return out
    
    def postprocess(self, outputs, img_shape, conf_thres=0.35, iou_thres=0.6, max_det=50):
        """
        后处理：将检测头输出转换为标准格式并应用 NMS
        
        Args:
            outputs: 检测头输出 (B, num_anchors, grid_h, grid_w, 5+num_classes)
            img_shape: 原始图像形状 (H, W)
            conf_thres: 置信度阈值，默认 0.35
            iou_thres: NMS IoU 阈值，默认 0.6
            max_det: 最大检测数量，默认 50
        
        Returns:
            标准格式的检测结果 (list of tensors)
            每个 tensor 形状：(N, 6) - (x1,y1,x2,y2,conf,cls)
        """
        import torch
        from torchvision.ops import batched_nms
        
        # 获取批次大小
        batch_size = outputs.shape[0]
        
        # 获取设备
        device = outputs.device
        
        # 结果列表
        results = []
        
        # YOLO 的 anchor 尺寸 (9 个 anchor，分为 3 组)
        # 来源：COCO 数据集统计
        # 第 1 组 (小目标): [10,13], [16,30], [33,23]
        # 第 2 组 (中目标): [30,61], [62,45], [59,119]
        # 第 3 组 (大目标): [116,90], [156,198], [373,326]
        anchors = torch.tensor([
            [10, 13], [16, 30], [33, 23],
            [30, 61], [62, 45], [59, 119],
            [116, 90], [156, 198], [373, 326]
        ], dtype=torch.float32, device=device)
        
        # 遍历批次中的每个样本
        for i in range(batch_size):
            # 获取当前图像的预测
            pred = outputs[i]
            
            # 获取维度信息
            num_anchors, grid_h, grid_w, _ = pred.shape
            
            # 展平空间维度
            # 从 (num_anchors, grid_h, grid_w, 5+num_classes) 
            # 到 (num_anchors*grid_h*grid_w, 5+num_classes)
            pred = pred.reshape(-1, 5 + self.num_classes)
            
            # 分离预测值
            # pred_boxes: 边界框参数 (x,y,w,h) - 前 4 列
            pred_boxes = pred[:, :4]
            
            # obj_conf: 目标置信度 - 第 5 列
            # sigmoid 激活：将 logits 转换为概率 (0-1)
            obj_conf = torch.sigmoid(pred[:, 4])
            
            # class_scores: 类别分数 - 第 6 列及以后
            # sigmoid 激活：将 logits 转换为概率
            class_scores = torch.sigmoid(pred[:, 5:])
            
            # 生成网格坐标
            # grid_y: [0, 1, 2, ..., grid_h-1]
            grid_y = torch.arange(grid_h, device=device)
            
            # grid_x: [0, 1, 2, ..., grid_w-1]
            grid_x = torch.arange(grid_w, device=device)
            
            # meshgrid: 创建网格
            # indexing='ij': 矩阵索引方式
            grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
            
            # stack: 堆叠为 (grid_h, grid_w, 2)
            grid = torch.stack((grid_x, grid_y), dim=-1)
            
            # reshape: 展平为 (grid_h*grid_w, 2)
            grid = grid.reshape(-1, 2).float()
            
            # repeat: 重复 num_anchors 次
            # 因为每个网格点有 num_anchors 个预测
            grid = grid.repeat(num_anchors, 1)
            
            # 选择对应的 anchor
            # anchor_idx: [0,0,0, 1,1,1, ..., num_anchors-1,num_anchors-1,num_anchors-1]
            anchor_idx = torch.arange(num_anchors, device=device).view(-1, 1, 1)
            anchor_idx = anchor_idx.repeat(1, grid_h, grid_w).reshape(-1)
            
            # 选择 anchor (循环使用 9 个 anchor)
            selected_anchors = anchors[anchor_idx % len(anchors)]
            
            # 解码边界框
            # stride: 网格步长 = 图像宽度 / 网格宽度
            stride = img_shape[1] / grid_w
            
            # cx, cy: 中心坐标
            # sigmoid(tx) + grid_offset
            cx = (torch.sigmoid(pred_boxes[:, 0]) + grid[:, 0]) * stride
            cy = (torch.sigmoid(pred_boxes[:, 1]) + grid[:, 1]) * stride
            
            # w, h: 宽高
            # exp(tw,th) * anchor_w,anchor_h
            w = torch.exp(pred_boxes[:, 2]) * selected_anchors[:, 0]
            h = torch.exp(pred_boxes[:, 3]) * selected_anchors[:, 1]
            
            # 转换为 [x1, y1, x2, y2] 格式
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            boxes = torch.stack((x1, y1, x2, y2), dim=1)
            
            # 计算最终置信度
            # class_conf: 最高类别分数
            # class_pred: 预测类别 ID
            class_conf, class_pred = torch.max(class_scores, dim=1)
            
            # conf: 最终置信度 = 目标置信度 * 类别置信度
            conf = obj_conf * class_conf
            
            # 应用置信度阈值
            # 只保留置信度 > conf_thres 的检测框
            mask = conf > conf_thres
            boxes = boxes[mask]
            conf = conf[mask]
            class_pred = class_pred[mask]
            
            # 处理没有检测框的情况
            if len(boxes) == 0:
                results.append(torch.zeros((0, 6), device=device))
                continue
            
            # 按置信度排序，取前 max_det 个
            conf_sort_idx = torch.argsort(conf, descending=True)
            if len(conf_sort_idx) > max_det:
                conf_sort_idx = conf_sort_idx[:max_det]
            boxes = boxes[conf_sort_idx]
            conf = conf[conf_sort_idx]
            class_pred = class_pred[conf_sort_idx]
            
            # 应用 NMS (非极大值抑制)
            # 去除重叠的边界框，保留置信度最高的
            from torchvision.ops import nms
            keep = nms(boxes, conf, iou_thres)
            
            # 获取最终结果
            boxes = boxes[keep]
            conf = conf[keep]
            class_pred = class_pred[keep]
            
            # 组合结果
            # 格式：(x1, y1, x2, y2, conf, cls)
            result = torch.cat((
                boxes, 
                conf.unsqueeze(1), 
                class_pred.unsqueeze(1).float()
            ), dim=1)
            results.append(result)
        
        return results

class YOLOTransformerLowLight(nn.Module):
    """
    低光照视频目标检测增强模型
    整合 YOLO26n + 低光增强 + Transformer 融合
    
    架构:
    输入 → 低光增强 → Backbone → 特征融合 → 检测头 → 输出
              ↓           ↑
         高清特征缓存
    """
    
    def __init__(self, model_path='yolo26n.pt'):
        """
        初始化模型
        
        Args:
            model_path: 预训练 YOLO 模型路径
        """
        super(YOLOTransformerLowLight, self).__init__()
        
        # 加载 YOLO26n 模型
        self.yolo = YOLO(model_path)
        
        # 获取 YOLO 模型的 Backbone
        # YOLO 模型结构：model[0]=Backbone, model[1]=Neck, model[2]=Head
        self.backbone = self.yolo.model.model[0]
        
        # 冻结 Backbone 参数
        # 原因：Backbone 已经预训练好，不需要更新
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 初始化低光增强模块
        self.light_enhance = LightEnhance()
        
        # 初始化 Transformer 融合模块
        self.fusion = MobileViTAttention()
        
        # 特征压缩层 (1x1 卷积)
        # 作用：将低光特征压缩到配置维度 (128)
        self.feature_compress = nn.Conv2d(16, Config.FEATURE_DIM, kernel_size=1)
        
        # 自定义检测头
        self.custom_head = CustomDetectHead(num_classes=80)
        
        # 损失权重
        self.dark_loss_weight = Config.DARK_LOSS_WEIGHT  # 1.8
        self.alignment_loss_weight = Config.ALIGNMENT_LOSS_WEIGHT  # 0.5
    
    def get_dark_mask(self, image):
        """
        生成暗部掩码
        
        Args:
            image: 输入图像 (B, 3, H, W)
        
        Returns:
            暗部掩码 (B, 1, H, W)
            1 表示暗部 (灰度<50), 0 表示亮部
        """
        # RGB 转灰度 - 向量化操作 (快速)
        # 使用标准灰度系数
        gray = image[:, 0] * 0.299 + image[:, 1] * 0.587 + image[:, 2] * 0.114
        
        # 生成二值掩码
        # gray < DARK_THRESHOLD: bool 张量
        # .float(): 转换为 float
        # .unsqueeze(1): 增加通道维度
        mask = (gray < Config.DARK_THRESHOLD).float().unsqueeze(1)
        
        return mask
    
    def forward(self, low_light_frame, high_res_feat=None, is_training=False):
        """
        前向传播
        
        Args:
            low_light_frame: 低光帧 (B, 3, H, W)
            high_res_feat: 高清缓存特征 (B, 128) 或 None
            is_training: 是否为训练模式
        
        Returns:
            检测结果
        """
        import time
        
        try:
            # 1. 低光增强
            enhance_start = time.time()
            enhanced_frame = self.light_enhance(low_light_frame)
            enhance_time = time.time() - enhance_start
            
            # 2. Backbone 提取特征
            backbone_start = time.time()
            low_light_feat = self.backbone(enhanced_frame)
            backbone_time = time.time() - backbone_start
            
            # 3. 生成暗部掩码
            mask_start = time.time()
            dark_mask = self.get_dark_mask(low_light_frame)
            mask_time = time.time() - mask_start
            
            # 调整暗部掩码尺寸以匹配特征图
            dark_mask = F.interpolate(
                dark_mask, 
                size=(low_light_feat.shape[2], low_light_feat.shape[3]), 
                mode='bilinear', 
                align_corners=False
            )
            
            # 4. 特征融合
            fusion_start = time.time()
            if high_res_feat is not None:
                # 有高清特征，进行融合
                fused_feat = self.fusion(low_light_feat, high_res_feat, dark_mask)
            else:
                # 无高清特征，使用原始特征
                fused_feat = low_light_feat
            fusion_time = time.time() - fusion_start
            
            # 5. 检测头输出
            head_start = time.time()
            outputs = self.custom_head(fused_feat)
            head_time = time.time() - head_start
            
            # 时间统计
            total_time = enhance_time + backbone_time + mask_time + fusion_time + head_time
            
            print(f"  Forward 时间分解:")
            print(f"    - 低光增强：{enhance_time*1000:.1f}ms ({enhance_time/total_time*100:.1f}%)")
            print(f"    - Backbone 特征提取：{backbone_time*1000:.1f}ms ({backbone_time/total_time*100:.1f}%)")
            print(f"    - 暗部掩码生成：{mask_time*1000:.1f}ms ({mask_time/total_time*100:.1f}%)")
            print(f"    - 特征融合：{fusion_time*1000:.1f}ms ({fusion_time/total_time*100:.1f}%)")
            print(f"    - 检测头：{head_time*1000:.1f}ms ({head_time/total_time*100:.1f}%)")
            
            return outputs
        except Exception as e:
            # 错误处理
            print(f"前向传播失败：{e}")
            import traceback
            traceback.print_exc()
            
            # 回退到不使用融合
            enhanced_frame = self.light_enhance(low_light_frame)
            low_light_feat = self.backbone(enhanced_frame)
            outputs = self.custom_head(low_light_feat)
            return outputs
    
    def loss_fn(self, outputs, targets, low_light_feat, high_res_feat):
        """
        自定义损失函数 - YOLO 风格
        
        Args:
            outputs: 模型输出 [batch, num_anchors, grid_h, grid_w, 5+num_classes]
            targets: 真实标签 (list of dicts)
            low_light_feat: 低光特征
            high_res_feat: 高清特征
        
        Returns:
            损失字典
        """
        # 初始化损失
        total_loss = 0
        box_loss = torch.tensor(0.0, device=outputs.device)
        obj_loss = torch.tensor(0.0, device=outputs.device)
        cls_loss = torch.tensor(0.0, device=outputs.device)
        alignment_loss = torch.tensor(0.0, device=outputs.device)
        
        # YOLO 损失权重
        box_gain = 0.05  # 定位损失权重
        cls_gain = 0.5   # 分类损失权重
        obj_gain = 1.0   # 置信度损失权重
        
        # 处理每个 batch
        if targets and len(targets) > 0:
            batch_size = outputs.shape[0]
            num_classes = self.custom_head.num_classes
            
            for batch_idx in range(batch_size):
                # 获取预测
                pred = outputs[batch_idx]
                
                # 展平
                pred_flat = pred.view(-1, 5 + num_classes)
                
                # 分离预测值
                pred_boxes = pred_flat[:, :4]  # [x, y, w, h]
                pred_obj = pred_flat[:, 4]     # 置信度
                pred_cls = pred_flat[:, 5:]    # 类别
                
                # 获取标签
                try:
                    target_list = targets[batch_idx]
                except:
                    continue
                
                if not target_list:
                    continue
                
                # 解析标签
                if isinstance(target_list, list) and len(target_list) > 0:
                    target_cls_list = []
                    target_cxywh_list = []
                    
                    for t in target_list:
                        if 'cls' in t and 'bboxes' in t:
                            cls = t['cls']
                            bboxes = t['bboxes']
                            if len(bboxes) == 4:
                                target_cls_list.append(cls)
                                target_cxywh_list.append(bboxes)
                    
                    if len(target_cls_list) == 0:
                        continue
                    
                    target_cls = torch.tensor(target_cls_list).long()
                    target_cxywh = torch.tensor(target_cxywh_list)
                else:
                    continue
                
                # 转换为网格坐标
                grid_h, grid_w = pred.shape[0], pred.shape[1]
                target_cx = target_cxywh[:, 0] * grid_w
                target_cy = target_cxywh[:, 1] * grid_h
                target_w = target_cxywh[:, 2] * grid_w
                target_h = target_cxywh[:, 3] * grid_h
                
                # 计算损失
                for i in range(len(target)):
                    # 找到最近的 k 个预测框
                    cx_diff = torch.abs(pred_boxes[:, 0] - target_cx[i])
                    cy_diff = torch.abs(pred_boxes[:, 1] - target_cy[i])
                    dist = cx_diff + cy_diff
                    
                    k = min(10, len(pred_boxes))
                    _, topk_idx = torch.topk(dist, k, largest=False)
                    
                    # 1. 定位损失 (MSE)
                    box_loss += F.mse_loss(pred_boxes[topk_idx, 0], target_cx[i].expand(k)) * box_gain
                    box_loss += F.mse_loss(pred_boxes[topk_idx, 1], target_cy[i].expand(k)) * box_gain
                    box_loss += F.mse_loss(pred_boxes[topk_idx, 2], target_w[i].expand(k)) * box_gain
                    box_loss += F.mse_loss(pred_boxes[topk_idx, 3], target_h[i].expand(k)) * box_gain
                    
                    # 2. 置信度损失 (BCE)
                    obj_target = torch.ones(k, device=outputs.device)
                    obj_loss += F.binary_cross_entropy_with_logits(pred_obj[topk_idx], obj_target) * obj_gain
                    
                    # 3. 分类损失 (BCE)
                    cls_target = torch.zeros_like(pred_cls[topk_idx])
                    cls_target[torch.arange(k), target_cls[i]] = 1.0
                    cls_loss += F.binary_cross_entropy_with_logits(pred_cls[topk_idx], cls_target) * cls_gain
            
            # 平均损失
            num_targets = sum(len(t) for t in targets if t)
            if num_targets > 0:
                box_loss = box_loss / num_targets
                obj_loss = obj_loss / num_targets
                cls_loss = cls_loss / num_targets
        
        # 总检测损失
        detection_loss = box_loss + obj_loss + cls_loss
        total_loss += detection_loss
        
        # 特征对齐损失
        if high_res_feat is not None:
            # 压缩低光特征
            low_light_feat_compressed = self.feature_compress(low_light_feat)
            
            # 全局池化
            low_light_feat_global = F.adaptive_avg_pool2d(
                low_light_feat_compressed, (1, 1)
            ).squeeze()
            
            # MSE 对齐损失
            alignment_loss = F.mse_loss(low_light_feat_global, high_res_feat) * self.alignment_loss_weight
            total_loss += alignment_loss
        
        return {
            'total_loss': total_loss,
            'box_loss': box_loss.item(),
            'obj_loss': obj_loss.item(),
            'cls_loss': cls_loss.item(),
            'alignment_loss': alignment_loss.item()
        }
    
    def detect(self, image, high_res_feat=None):
        """
        推理方法
        
        Args:
            image: 输入图像 (numpy 数组或张量)
            high_res_feat: 高清缓存特征
        
        Returns:
            检测结果 (ultralytics Results 格式)
        """
        import time
        total_start = time.time()
        
        with torch.no_grad():
            # 1. 预处理
            preproc_start = time.time()
            if isinstance(image, np.ndarray):
                # numpy → tensor
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
                image_tensor = image_tensor.to(next(self.parameters()).device)
            else:
                image_tensor = image
            
            # 高清特征设备同步
            if high_res_feat is not None:
                high_res_feat = high_res_feat.to(next(self.parameters()).device)
            preproc_time = time.time() - preproc_start
            
            # 2. 前向传播
            forward_start = time.time()
            print("使用融合特征进行前向传播")
            outputs = self.forward(image_tensor, high_res_feat, is_training=False)
            forward_time = time.time() - forward_start
            
            # 3. 后处理
            postproc_start = time.time()
            print("使用自定义后处理方法处理融合特征输出")
            img_shape = (image_tensor.shape[2], image_tensor.shape[3])
            results = self.custom_head.postprocess(outputs, img_shape)
            postproc_time = time.time() - postproc_start
            
            # 4. 结果转换
            convert_start = time.time()
            from ultralytics.engine.results import Results
            results_obj = Results(
                path='image0.jpg',
                boxes=torch.cat(results, dim=0) if len(results) > 0 and results[0].shape[0] > 0 else torch.zeros((0, 6)),
                orig_img=image_tensor.squeeze().permute(1, 2, 0).cpu().numpy(),
                names={i: f'class_{i}' for i in range(80)}
            )
            convert_time = time.time() - convert_start
            
            # 时间统计
            total_time = time.time() - total_start
            print(f"\n增强 YOLO 推理时间分解:")
            print(f"  1. 预处理：{preproc_time*1000:.1f}ms")
            print(f"  2. 前向传播：{forward_time*1000:.1f}ms")
            print(f"  3. 后处理：{postproc_time*1000:.1f}ms")
            print(f"  4. 结果转换：{convert_time*1000:.1f}ms")
            print(f"  总耗时：{total_time*1000:.1f}ms")
            
            return results_obj
```
