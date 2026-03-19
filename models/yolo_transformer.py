# 核心检测模型
# 整合 YOLO26n Backbone + LightEnhance + MobileViTAttention + 自定义检测头

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from ultralytics import YOLO
from light_enhance import LightEnhance
from fusion_module import MobileViTAttention
from config import Config

class CustomDetectHead(nn.Module):
    """
    自定义检测头 - 行人检测优化版
    针对单类别（行人）检测优化，大幅减少参数量和计算量
    
    优化点:
    1. num_classes 从 80 减少到 1 (只检测行人)
    2. 输出通道从 (5+80)*3=255 减少到 (5+1)*3=18
    3. 参数量减少约 14 倍
    4. 推理速度提升
    """
    def __init__(self, num_classes=1):  # 只检测行人
        super(CustomDetectHead, self).__init__()
        self.num_classes = num_classes  # 1 (行人)
        
        # 检测层：输出边界框、置信度和类别
        # 输入：16 通道
        # 输出：(5 + num_classes) * 3 = (5+1)*3 = 18 通道
        #   - 5: 边界框 4 个参数 (x,y,w,h) + 1 个目标置信度
        #   - 1: 类别概率 (行人)
        #   - 3: 3 个 anchor
        self.detect = nn.Conv2d(16, (5 + num_classes) * 3, kernel_size=1)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 融合特征 (B, 16, H, W)
            
        Returns:
            检测结果 (B, 3, H, W, 6)
            - 3: 3 个 anchor
            - 6: 4 个坐标 + 1 个置信度 + 1 个类别
        """
        # 确保特征连续
        x = x.contiguous()
        
        # 通过检测层
        out = self.detect(x)
        
        # 调整输出格式
        batch_size = out.shape[0]
        
        # 将输出转换为 [batch, num_anchors, grid_h, grid_w, 5+num_classes]
        # 对于行人检测：[batch, 3, H, W, 6]
        out = out.view(batch_size, 3, 6, -1, out.shape[-1])
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        
        return out
    
    def postprocess(self, outputs, img_shape, conf_thres=0.35, iou_thres=0.6, max_det=50):
        """
        后处理方法：将检测头输出转换为标准格式并应用 NMS（行人检测优化版）
        
        Args:
            outputs: 检测头输出 [batch, num_anchors, grid_h, grid_w, 6]
            img_shape: 原始图像形状 (H, W)
            conf_thres: 置信度阈值
            iou_thres: NMS IoU 阈值
            max_det: 最大检测数量
            
        Returns:
            标准格式的检测结果
        """
        import torch
        from torchvision.ops import nms
        
        # 获取批次大小
        batch_size = outputs.shape[0]
        device = outputs.device
        results = []
        
        # 假设的 anchor 尺寸
        anchors = torch.tensor([
            [10, 13], [16, 30], [33, 23],
            [30, 61], [62, 45], [59, 119],
            [116, 90], [156, 198], [373, 326]
        ], dtype=torch.float32, device=device)
        
        for i in range(batch_size):
            # 获取当前图像的预测
            pred = outputs[i]
            num_anchors, grid_h, grid_w, _ = pred.shape
            
            # 展平空间维度
            pred = pred.reshape(-1, 6)  # 行人检测：5+1=6
            
            # 分离边界框和置信度、类别
            pred_boxes = pred[:, :4]
            obj_conf = torch.sigmoid(pred[:, 4])
            class_score = torch.sigmoid(pred[:, 5])  # 行人检测：只有 1 个类别
            
            # 生成网格坐标
            grid_y = torch.arange(grid_h, device=device)
            grid_x = torch.arange(grid_w, device=device)
            grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
            grid = torch.stack((grid_x, grid_y), dim=-1)
            grid = grid.reshape(-1, 2).float()
            grid = grid.repeat(num_anchors, 1)
            
            # 选择对应的 anchor
            anchor_idx = torch.arange(num_anchors, device=device).view(-1, 1, 1)
            anchor_idx = anchor_idx.repeat(1, grid_h, grid_w).reshape(-1)
            selected_anchors = anchors[anchor_idx % len(anchors)]
            
            # 解码边界框
            stride = img_shape[1] / grid_w
            cx = (torch.sigmoid(pred_boxes[:, 0]) + grid[:, 0]) * stride
            cy = (torch.sigmoid(pred_boxes[:, 1]) + grid[:, 1]) * stride
            w = torch.exp(pred_boxes[:, 2]) * selected_anchors[:, 0]
            h = torch.exp(pred_boxes[:, 3]) * selected_anchors[:, 1]
            
            # 转换为 [x1, y1, x2, y2] 格式
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            boxes = torch.stack((x1, y1, x2, y2), dim=1)
            
            # 计算最终置信度（行人检测简化版）
            # 单类别情况下，class_score 就是行人置信度
            conf = obj_conf * class_score
            class_pred = torch.zeros(len(conf), dtype=torch.long, device=device)  # 行人类别 ID=0
            
            # 应用置信度阈值
            mask = conf > conf_thres
            boxes = boxes[mask]
            conf = conf[mask]
            class_pred = class_pred[mask]
            
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
            
            # 应用 NMS
            keep = nms(boxes, conf, iou_thres)
            
            # 获取最终结果
            boxes = boxes[keep]
            conf = conf[keep]
            class_pred = class_pred[keep]
            
            # 组合结果
            result = torch.cat((boxes, conf.unsqueeze(1), class_pred.unsqueeze(1).float()), dim=1)
            results.append(result)
        
        return results

class YOLOTransformerLowLight(nn.Module):
    """
    低光照视频目标检测增强模型
    整合 YOLO26n + 低光增强 + Transformer 融合
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
        self.backbone = self.yolo.model.model[0]
        
        # 冻结 Backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 初始化低光增强模块
        self.light_enhance = LightEnhance()
        
        # 初始化 Transformer 融合模块
        self.fusion = MobileViTAttention()
        
        # 特征压缩层（用于高清特征）
        self.feature_compress = nn.Conv2d(16, Config.FEATURE_DIM, kernel_size=1)
        
        # 自定义检测头（行人检测优化）
        self.custom_head = CustomDetectHead(num_classes=1)
        
        # 损失权重
        self.dark_loss_weight = Config.DARK_LOSS_WEIGHT
        self.alignment_loss_weight = Config.ALIGNMENT_LOSS_WEIGHT
    
    def get_dark_mask(self, image):
        """
        生成暗部掩码（灰度 < 50 为 1）
        
        Args:
            image: 输入图像 (B, 3, H, W)
            
        Returns:
            暗部掩码 (B, 1, H, W)
        """
        # 使用更快的灰度转换方法（向量化操作）
        # 预计算的灰度系数
        gray = image[:, 0] * 0.299 + image[:, 1] * 0.587 + image[:, 2] * 0.114
        # 生成掩码
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
            
            # 2. 通过 Backbone 提取特征
            backbone_start = time.time()
            low_light_feat = self.backbone(enhanced_frame)
            backbone_time = time.time() - backbone_start
            
            # 3. 生成暗部掩码
            mask_start = time.time()
            dark_mask = self.get_dark_mask(low_light_frame)
            mask_time = time.time() - mask_start
            
            # 调整暗部掩码的尺寸以匹配 low_light_feat
            dark_mask = F.interpolate(dark_mask, size=(low_light_feat.shape[2], low_light_feat.shape[3]), mode='bilinear', align_corners=False)
            
            # 4. 特征融合
            fusion_start = time.time()
            # 如果有高清特征且不是训练模式，进行融合
            if high_res_feat is not None:
                # 融合特征
                fused_feat = self.fusion(low_light_feat, high_res_feat, dark_mask)
            else:
                # 使用原始低光特征
                fused_feat = low_light_feat
            fusion_time = time.time() - fusion_start
            
            # 5. 通过自定义检测头
            head_start = time.time()
            outputs = self.custom_head(fused_feat)
            head_time = time.time() - head_start
            
            total_time = enhance_time + backbone_time + mask_time + fusion_time + head_time
            
            print(f"  Forward 时间分解:")
            print(f"    - 低光增强：{enhance_time*1000:.1f}ms ({enhance_time/total_time*100:.1f}%)")
            print(f"    - Backbone 特征提取：{backbone_time*1000:.1f}ms ({backbone_time/total_time*100:.1f}%)")
            print(f"    - 暗部掩码生成：{mask_time*1000:.1f}ms ({mask_time/total_time*100:.1f}%)")
            print(f"    - 特征融合：{fusion_time*1000:.1f}ms ({fusion_time/total_time*100:.1f}%)")
            print(f"    - 检测头：{head_time*1000:.1f}ms ({head_time/total_time*100:.1f}%)")
            
            return outputs
        except Exception as e:
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
            targets: 真实标签 (list of list of dicts, 每个 dict 包含 bbox 和 class)
            low_light_feat: 低光特征
            high_res_feat: 高清特征
            
        Returns:
            总损失
        """
        total_loss = 0
        box_loss = torch.tensor(0.0, device=outputs.device)
        obj_loss = torch.tensor(0.0, device=outputs.device)
        cls_loss = torch.tensor(0.0, device=outputs.device)
        alignment_loss = torch.tensor(0.0, device=outputs.device)
        
        # YOLO 损失参数
        box_gain = 0.05  # 定位损失权重
        cls_gain = 0.5   # 分类损失权重
        obj_gain = 1.0   # 置信度损失权重
        
        if targets and len(targets) > 0:
            batch_size = outputs.shape[0]
            num_classes = self.custom_head.num_classes
            
            # 处理每个 batch
            for batch_idx in range(batch_size):
                # 获取当前 batch 的预测
                pred = outputs[batch_idx]  # [num_anchors, grid_h, grid_w, 5+num_classes]
                
                # 展平空间维度
                pred_flat = pred.view(-1, 5 + num_classes)  # [num_anchors * grid_h * grid_w, 5+num_classes]
                
                # 分离预测值
                pred_boxes = pred_flat[:, :4]  # [x, y, w, h]
                pred_obj = pred_flat[:, 4]     # 目标置信度
                pred_cls = pred_flat[:, 5:]    # 类别分数
                
                # 获取当前 batch 的真实标签
                try:
                    target_list = targets[batch_idx]  # list of dicts or list of lists
                except (IndexError, TypeError, KeyError):
                    continue
                
                if target_list is None or (isinstance(target_list, (list, tuple)) and len(target_list) == 0):
                    continue
                
                # targets 的格式是：[{'img_idx': i, 'cls': cls, 'bboxes': bboxes}, ...]
                # 我们需要从这些 dict 中提取标签
                if isinstance(target_list, list) and len(target_list) > 0 and isinstance(target_list[0], dict):
                    # 从 dict 中提取 cls 和 bboxes
                    try:
                        target_cls_list = []
                        target_cxywh_list = []
                        for t in target_list:
                            if 'cls' in t and 'bboxes' in t:
                                cls = t['cls']
                                bboxes = t['bboxes']
                                # bboxes 格式是 [x, y, w, h]（已经是归一化的）
                                if len(bboxes) == 4:
                                    target_cls_list.append(cls)
                                    target_cxywh_list.append(bboxes)
                        
                        if len(target_cls_list) == 0:
                            continue
                        
                        target_cls = torch.tensor(target_cls_list).long()
                        target_cxywh = torch.tensor(target_cxywh_list)
                    except Exception as e:
                        print(f"处理 dict 格式标签失败：{e}")
                        continue
                else:
                    # 跳过其他格式的标签
                    continue
                
                if target is None or len(target) == 0:
                    continue
                
                # 将真实标签转换为相对于网格的坐标
                grid_h, grid_w = pred.shape[0], pred.shape[1]
                
                # 真实框已经是归一化的中心坐标和宽高
                target_cx = target_cxywh[:, 0] * grid_w
                target_cy = target_cxywh[:, 1] * grid_h
                target_w = target_cxywh[:, 2] * grid_w
                target_h = target_cxywh[:, 3] * grid_h
                
                # 为每个真实框计算损失
                for i in range(len(target_cls)):
                    # 找到距离真实框中心最近的预测框
                    cx_diff = torch.abs(pred_boxes[:, 0] - target_cx[i])
                    cy_diff = torch.abs(pred_boxes[:, 1] - target_cy[i])
                    dist = cx_diff + cy_diff
                    
                    # 选择最近的 k 个预测框
                    k = min(10, len(pred_boxes))
                    _, topk_idx = torch.topk(dist, k, largest=False)
                    
                    # 1. 定位损失 (MSE)
                    box_loss += F.mse_loss(pred_boxes[topk_idx, 0], target_cx[i].expand(k)) * box_gain
                    box_loss += F.mse_loss(pred_boxes[topk_idx, 1], target_cy[i].expand(k)) * box_gain
                    box_loss += F.mse_loss(pred_boxes[topk_idx, 2], target_w[i].expand(k)) * box_gain
                    box_loss += F.mse_loss(pred_boxes[topk_idx, 3], target_h[i].expand(k)) * box_gain
                    
                    # 2. 置信度损失 (BCE) - 正样本
                    obj_target = torch.ones(k, device=outputs.device)
                    obj_loss += F.binary_cross_entropy_with_logits(pred_obj[topk_idx], obj_target) * obj_gain
                    
                    # 3. 分类损失 (BCE) - 行人检测优化
                    # 单类别情况下，target_cls[i] 应该始终为 0（行人）
                    cls_target = torch.zeros_like(pred_cls[topk_idx])
                    cls_target[:, 0] = 1.0  # 行人类别
                    cls_loss += F.binary_cross_entropy_with_logits(pred_cls[topk_idx], cls_target) * cls_gain
            
            # 平均损失
            num_targets = sum(len(t) for t in targets if t is not None and len(t) > 0)
            if num_targets > 0:
                box_loss = box_loss / num_targets
                obj_loss = obj_loss / num_targets
                cls_loss = cls_loss / num_targets
        
        # 总检测损失
        detection_loss = box_loss + obj_loss + cls_loss
        total_loss += detection_loss
        
        # 跨特征对齐损失
        if high_res_feat is not None:
            # 压缩低光特征到相同维度
            low_light_feat_compressed = self.feature_compress(low_light_feat)
            # 全局池化
            low_light_feat_global = F.adaptive_avg_pool2d(low_light_feat_compressed, (1, 1)).squeeze()
            # 计算对齐损失
            alignment_loss = F.mse_loss(low_light_feat_global, high_res_feat) * self.alignment_loss_weight
            total_loss += alignment_loss
        
        return {
            'total_loss': total_loss,
            'box_loss': box_loss.item() if isinstance(box_loss, torch.Tensor) else box_loss,
            'obj_loss': obj_loss.item() if isinstance(obj_loss, torch.Tensor) else obj_loss,
            'cls_loss': cls_loss.item() if isinstance(cls_loss, torch.Tensor) else cls_loss,
            'alignment_loss': alignment_loss.item() if isinstance(alignment_loss, torch.Tensor) else alignment_loss
        }
    
    def detect(self, image, high_res_feat=None):
        """
        推理方法
        
        Args:
            image: 输入图像（弱光环境的照片，存在物体或人）
            high_res_feat: 高清缓存特征（当前场景的高清环境照片特征）
            
        Returns:
            检测结果
        """
        import time
        total_start = time.time()
        
        with torch.no_grad():
            # 1. 预处理
            preproc_start = time.time()
            if isinstance(image, np.ndarray):
                # 转换为张量并移动到正确的设备
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
                image_tensor = image_tensor.to(next(self.parameters()).device)
            else:
                image_tensor = image
            
            # 确保高清特征在正确的设备上
            if high_res_feat is not None:
                high_res_feat = high_res_feat.to(next(self.parameters()).device)
            preproc_time = time.time() - preproc_start
            
            # 2. 前向传播（包含低光增强、特征提取、特征融合、检测头）
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
            
            # 3. 结果转换
            convert_start = time.time()
            # 将结果转换为 ultralytics 格式（行人检测优化）
            from ultralytics.engine.results import Results
            results_obj = Results(
                path='image0.jpg',
                boxes=torch.cat(results, dim=0) if len(results) > 0 and results[0].shape[0] > 0 else torch.zeros((0, 6)),
                orig_img=image_tensor.squeeze().permute(1, 2, 0).cpu().numpy(),
                names={0: 'person'}  # 只检测行人
            )
            convert_time = time.time() - convert_start
            
            total_time = time.time() - total_start
            
            # 打印时间统计
            print(f"\n增强 YOLO 推理时间分解:")
            print(f"  1. 预处理（图像转张量）: {preproc_time*1000:.1f}ms ({preproc_time/total_time*100:.1f}%)")
            print(f"  2. 前向传播（增强 + 融合 + 检测头）: {forward_time*1000:.1f}ms ({forward_time/total_time*100:.1f}%)")
            print(f"  3. 后处理（NMS 等）: {postproc_time*1000:.1f}ms ({postproc_time/total_time*100:.1f}%)")
            print(f"  4. 结果转换: {convert_time*1000:.1f}ms ({convert_time/total_time*100:.1f}%)")
            print(f"  总耗时：{total_time*1000:.1f}ms")
            print()
            
            return results_obj
